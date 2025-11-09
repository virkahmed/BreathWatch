#!/usr/bin/env python3
"""
Train a multi-task CNN for cough classification and attribute prediction on COUGHVID tiles.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Try to import scikit-learn metrics
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. AUC metrics will be skipped.", file=sys.stderr)
    print("Install with: pip install scikit-learn", file=sys.stderr)


class CoughMultitaskDataset(Dataset):
    """Dataset for multi-task cough classification and attribute prediction."""
    
    def __init__(self, manifest_df: pd.DataFrame, attr_cols: List[str]):
        """
        Args:
            manifest_df: DataFrame with columns: path_npy, label, attr_<name>, attr_<name>_mask
            attr_cols: List of attribute names (e.g., ['wet', 'wheezing', ...])
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.attr_cols = attr_cols
    
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        
        # Load spectrogram
        spec = np.load(row['path_npy']).astype(np.float32)
        
        # Normalize: (S_db + 80) / 80
        spec = (spec + 80.0) / 80.0
        
        # Add channel dimension: [1, n_mels, n_frames]
        spec = spec[np.newaxis, :, :]
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(spec)
        
        # Binary label: 1 if (dataset == "coughvid" and label == "cough") else 0
        dataset = row.get('dataset', '')
        label = row.get('label', '')
        binary_label = 1 if (dataset == 'coughvid' and label == 'cough') else 0
        binary_label_tensor = torch.tensor(binary_label, dtype=torch.long)
        
        # Attribute labels and masks
        # For rows where dataset != "coughvid", force all masks to 0
        is_coughvid = (dataset == 'coughvid')
        
        attr_labels = []
        attr_masks = []
        for attr in self.attr_cols:
            attr_col = f'attr_{attr}'
            mask_col = f'attr_{attr}_mask'
            
            if pd.notna(row.get(attr_col)):
                attr_labels.append(float(row[attr_col]))
            else:
                attr_labels.append(0.0)  # Placeholder, will be masked
            
            # Force mask to 0 if not coughvid
            if is_coughvid and pd.notna(row.get(mask_col)):
                attr_masks.append(float(row[mask_col]))
            else:
                attr_masks.append(0.0)
        
        attr_labels_tensor = torch.tensor(attr_labels, dtype=torch.float32)
        attr_masks_tensor = torch.tensor(attr_masks, dtype=torch.float32)
        
        return spec_tensor, binary_label_tensor, attr_labels_tensor, attr_masks_tensor


class CoughMultitaskCNN(nn.Module):
    """Multi-task CNN with two heads for cough classification and attribute prediction."""
    
    def __init__(self, num_attrs: int):
        super(CoughMultitaskCNN, self).__init__()
        
        # Backbone: 3 conv blocks
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Head A: Binary classification (cough vs no_cough)
        self.head_a = nn.Linear(64, 2)
        
        # Head B: Multi-label attribute prediction
        self.head_b = nn.Linear(64, num_attrs)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Head A: Binary classification
        logits_a = self.head_a(x)
        
        # Head B: Multi-label attributes
        logits_b = self.head_b(x)
        probs_b = self.sigmoid(logits_b)
        
        return logits_a, probs_b


def get_device():
    """Get the best available device (MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def compute_masked_bce_loss(pred, target, mask):
    """
    Compute masked BCE loss.
    Compute per-attribute over labeled rows only, then average over attributes.
    Formula: (BCE(sigmoid(headB), y_attr) * mask).sum() / mask.sum().clamp(min=1)
    """
    # BCE loss per sample per attribute: [batch_size, num_attrs]
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    
    # Apply mask: only compute loss on labeled samples
    masked_bce = bce * mask
    
    # Sum over all (samples, attributes) where mask==1, divide by total mask sum
    # This gives average loss per labeled sample-attribute pair
    loss = masked_bce.sum() / mask.sum().clamp(min=1.0)
    
    return loss


def train_epoch(model, dataloader, criterion_bin, optimizer, device, scaler, attr_weight, use_amp):
    """Train for one epoch with optional AMP."""
    model.train()
    total_loss_bin = 0.0
    total_loss_attr = 0.0
    total_loss = 0.0
    correct_bin = 0
    total_bin = 0
    
    for spec, binary_label, attr_labels, attr_masks in tqdm(dataloader, desc="Training", leave=False):
        spec = spec.to(device)
        binary_label = binary_label.to(device)
        attr_labels = attr_labels.to(device)
        attr_masks = attr_masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with optional AMP
        if use_amp:
            with autocast(device_type=device.type, dtype=torch.float16):
                logits_a, probs_b = model(spec)
                
                # Binary loss
                loss_bin = criterion_bin(logits_a, binary_label)
                
                # Attribute loss (masked BCE)
                # Compute per-attribute over labeled rows only, then average over attrs
                loss_attr = compute_masked_bce_loss(probs_b, attr_labels, attr_masks)
                
                # Total loss
                loss = loss_bin + attr_weight * loss_attr
            
            # Backward pass with scaler (only for CUDA)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits_a, probs_b = model(spec)
            
            # Binary loss
            loss_bin = criterion_bin(logits_a, binary_label)
            
            # Attribute loss (masked BCE)
            loss_attr = compute_masked_bce_loss(probs_b, attr_labels, attr_masks)
            
            # Total loss
            loss = loss_bin + attr_weight * loss_attr
            
            # Backward pass without scaler
            loss.backward()
            optimizer.step()
        
        # Statistics
        total_loss_bin += loss_bin.item()
        total_loss_attr += loss_attr.item()
        total_loss += loss.item()
        
        _, predicted = torch.max(logits_a.data, 1)
        total_bin += binary_label.size(0)
        correct_bin += (predicted == binary_label).sum().item()
    
    avg_loss_bin = total_loss_bin / len(dataloader)
    avg_loss_attr = total_loss_attr / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    accuracy_bin = 100.0 * correct_bin / total_bin
    
    return avg_loss_bin, avg_loss_attr, avg_loss, accuracy_bin


def validate(model, dataloader, criterion_bin, device, attr_cols, use_amp):
    """Validate and compute metrics."""
    model.eval()
    total_loss_bin = 0.0
    total_loss_attr = 0.0
    
    all_binary_labels = []
    all_binary_probs = []
    all_attr_labels = []
    all_attr_probs = []
    all_attr_masks = []
    
    with torch.no_grad():
        for spec, binary_label, attr_labels, attr_masks in tqdm(dataloader, desc="Validating", leave=False):
            spec = spec.to(device)
            binary_label = binary_label.to(device)
            attr_labels = attr_labels.to(device)
            attr_masks = attr_masks.to(device)
            
            # Forward pass with optional AMP
            if use_amp:
                with autocast(device_type=device.type, dtype=torch.float16):
                    logits_a, probs_b = model(spec)
                    
                    # Binary loss
                    loss_bin = criterion_bin(logits_a, binary_label)
                    
                    # Attribute loss
                    loss_attr = compute_masked_bce_loss(probs_b, attr_labels, attr_masks)
            else:
                logits_a, probs_b = model(spec)
                
                # Binary loss
                loss_bin = criterion_bin(logits_a, binary_label)
                
                # Attribute loss
                loss_attr = compute_masked_bce_loss(probs_b, attr_labels, attr_masks)
            
            total_loss_bin += loss_bin.item()
            total_loss_attr += loss_attr.item()
            
            # Collect predictions for metrics
            probs_a = torch.softmax(logits_a, dim=1)
            p_cough = probs_a[:, 1].cpu().numpy()
            
            all_binary_labels.extend(binary_label.cpu().numpy())
            all_binary_probs.extend(p_cough)
            all_attr_labels.append(attr_labels.cpu().numpy())
            all_attr_probs.append(probs_b.cpu().numpy())
            all_attr_masks.append(attr_masks.cpu().numpy())
    
    avg_loss_bin = total_loss_bin / len(dataloader)
    avg_loss_attr = total_loss_attr / len(dataloader)
    
    # Convert to numpy arrays
    all_binary_labels = np.array(all_binary_labels)
    all_binary_probs = np.array(all_binary_probs)
    all_attr_labels = np.concatenate(all_attr_labels, axis=0)
    all_attr_probs = np.concatenate(all_attr_probs, axis=0)
    all_attr_masks = np.concatenate(all_attr_masks, axis=0)
    
    # Compute metrics
    metrics = {
        'loss_bin': avg_loss_bin,
        'loss_attr': avg_loss_attr
    }
    
    # Binary metrics
    if SKLEARN_AVAILABLE:
        try:
            metrics['binary_pr_auc'] = average_precision_score(all_binary_labels, all_binary_probs)
            metrics['binary_roc_auc'] = roc_auc_score(all_binary_labels, all_binary_probs)
        except Exception as e:
            print(f"Warning: Error computing binary AUC metrics: {e}", file=sys.stderr)
            metrics['binary_pr_auc'] = None
            metrics['binary_roc_auc'] = None
    else:
        metrics['binary_pr_auc'] = None
        metrics['binary_roc_auc'] = None
    
    # Per-attribute metrics
    attr_metrics = {}
    for i, attr in enumerate(attr_cols):
        # Get labeled samples only
        labeled_mask = all_attr_masks[:, i] == 1
        if labeled_mask.sum() > 0:
            attr_labels_labeled = all_attr_labels[labeled_mask, i]
            attr_probs_labeled = all_attr_probs[labeled_mask, i]
            
            if SKLEARN_AVAILABLE:
                try:
                    pr_auc = average_precision_score(attr_labels_labeled, attr_probs_labeled)
                    attr_metrics[f'{attr}_pr_auc'] = pr_auc
                except Exception as e:
                    attr_metrics[f'{attr}_pr_auc'] = None
            else:
                attr_metrics[f'{attr}_pr_auc'] = None
        else:
            attr_metrics[f'{attr}_pr_auc'] = None
    
    # Micro and macro mAP
    if SKLEARN_AVAILABLE:
        try:
            # Micro mAP: flatten all labeled predictions
            labeled_flat = all_attr_masks.flatten() == 1
            if labeled_flat.sum() > 0:
                labels_flat = all_attr_labels.flatten()[labeled_flat]
                probs_flat = all_attr_probs.flatten()[labeled_flat]
                metrics['micro_map'] = average_precision_score(labels_flat, probs_flat)
            else:
                metrics['micro_map'] = None
            
            # Macro mAP: average per-attribute PR-AUC
            attr_pr_aucs = [v for v in attr_metrics.values() if v is not None]
            if len(attr_pr_aucs) > 0:
                metrics['macro_map'] = np.mean(attr_pr_aucs)
            else:
                metrics['macro_map'] = None
        except Exception as e:
            print(f"Warning: Error computing mAP metrics: {e}", file=sys.stderr)
            metrics['micro_map'] = None
            metrics['macro_map'] = None
    else:
        metrics['micro_map'] = None
        metrics['macro_map'] = None
    
    metrics['attr_metrics'] = attr_metrics
    
    return metrics


def compute_prevalences(manifest_df: pd.DataFrame, attr_cols: List[str]) -> dict:
    """Compute prevalence of each attribute (on labeled samples only)."""
    prevalences = {}
    coughvid_rows = manifest_df[manifest_df['dataset'] == 'coughvid']
    
    for attr in attr_cols:
        mask_col = f'attr_{attr}_mask'
        attr_col = f'attr_{attr}'
        
        if mask_col in coughvid_rows.columns:
            labeled_mask = coughvid_rows[mask_col] == 1
            if labeled_mask.sum() > 0:
                positive_count = (coughvid_rows.loc[labeled_mask, attr_col] == 1).sum()
                prevalence = positive_count / labeled_mask.sum()
                prevalences[attr] = float(prevalence)
            else:
                prevalences[attr] = 0.0
        else:
            prevalences[attr] = 0.0
    
    return prevalences


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-task CNN for cough classification and attribute prediction'
    )
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to coughvid_tiles_attrs.csv')
    parser.add_argument('--val-manifest', type=str, default=None,
                        help='Optional path to external validation manifest. If provided, train_df from --manifest and val_df from --val-manifest (no internal split)')
    parser.add_argument('--attr-cols', type=str, default='wet,wheezing,stridor,choking,congestion',
                        help='Comma-separated attribute columns (default: wet,wheezing,stridor,choking,congestion)')
    parser.add_argument('--out', type=str, default='models/cough_multitask.pt',
                        help='Output model path (default: models/cough_multitask.pt)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs (default: 4)')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    parser.add_argument('--attr-weight', type=float, default=0.5,
                        help='Weight for attribute loss (default: 0.5)')
    
    args = parser.parse_args()
    
    # Parse attribute columns
    attr_cols = [col.strip() for col in args.attr_cols.split(',')]
    print(f"Attribute columns: {attr_cols}")
    
    # Check if scikit-learn is available
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn is not installed. AUC metrics will be skipped.", file=sys.stderr)
        print("To install: pip install scikit-learn", file=sys.stderr)
        print()
    
    # Load train manifest
    print(f"Loading train manifest from {args.manifest}...")
    try:
        train_df = pd.read_csv(args.manifest, low_memory=False)
    except Exception as e:
        print(f"Error loading train manifest: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(train_df)} train rows")
    
    # Load val manifest if provided, otherwise split internally
    if args.val_manifest:
        print(f"Loading validation manifest from {args.val_manifest}...")
        try:
            val_df = pd.read_csv(args.val_manifest, low_memory=False)
        except Exception as e:
            print(f"Error loading validation manifest: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(val_df)} validation rows")
        print("Using external validation set (no internal split)")
    else:
        print("No --val-manifest provided, performing internal split...")
        # Create binary label for splitting
        train_df['_binary_label'] = (
            (train_df['dataset'] == 'coughvid') & (train_df['label'] == 'cough')
        ).astype(int)
        
        # Split into train/val (85/15 stratified on binary label)
        if SKLEARN_AVAILABLE:
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.15,
                stratify=train_df['_binary_label'],
                random_state=42
            )
        else:
            print("Warning: Using non-stratified split (scikit-learn not available)", file=sys.stderr)
            train_size = int(0.85 * len(train_df))
            val_df = train_df.iloc[train_size:].copy()
            train_df = train_df.iloc[:train_size].copy()
    
    # Check required columns
    required_cols = ['path_npy', 'label', 'dataset']
    for attr in attr_cols:
        required_cols.extend([f'attr_{attr}', f'attr_{attr}_mask'])
    
    for df_name, df in [('train', train_df), ('val', val_df)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in {df_name} manifest: {missing_cols}", file=sys.stderr)
            sys.exit(1)
    
    # Fill missing mask columns with zeros
    for attr in attr_cols:
        mask_col = f'attr_{attr}_mask'
        for df in [train_df, val_df]:
            if mask_col not in df.columns:
                df[mask_col] = 0
            else:
                df[mask_col] = df[mask_col].fillna(0)
    
    # For rows with dataset != "coughvid", force all attr_*_mask = 0
    for df in [train_df, val_df]:
        non_coughvid_mask = df['dataset'] != 'coughvid'
        for attr in attr_cols:
            mask_col = f'attr_{attr}_mask'
            df.loc[non_coughvid_mask, mask_col] = 0
    
    # Create binary label: 1 if (dataset == "coughvid" and label == "cough") else 0
    train_df['_binary_label'] = (
        (train_df['dataset'] == 'coughvid') & (train_df['label'] == 'cough')
    ).astype(int)
    val_df['_binary_label'] = (
        (val_df['dataset'] == 'coughvid') & (val_df['label'] == 'cough')
    ).astype(int)
    
    # Print final train/val class counts before training
    train_n_pos = int((train_df['dataset'] == 'coughvid') & (train_df['label'] == 'cough')).sum()
    train_n_neg = len(train_df) - train_n_pos
    val_n_pos = int((val_df['dataset'] == 'coughvid') & (val_df['label'] == 'cough')).sum()
    val_n_neg = len(val_df) - val_n_pos
    
    print(f"\nFinal class counts before training:")
    print(f"  Train: {len(train_df)} samples (pos: {train_n_pos}, neg: {train_n_neg})")
    print(f"  Val: {len(val_df)} samples (pos: {val_n_pos}, neg: {val_n_neg})")
    print()
    
    # Create datasets
    train_dataset = CoughMultitaskDataset(train_df, attr_cols)
    val_dataset = CoughMultitaskDataset(val_df, attr_cols)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    device = get_device()
    print(f"Using device: {device}")
    model = CoughMultitaskCNN(num_attrs=len(attr_cols)).to(device)
    
    # --- class counts (train set) ---
    cond_pos = (train_df["dataset"].eq("coughvid") & train_df["label"].eq("cough"))
    train_n_pos = int(cond_pos.sum())
    train_n_neg = int(len(train_df) - train_n_pos)
    print(f"Train class counts -> pos: {train_n_pos} | neg: {train_n_neg}")
    
    # --- inverse-frequency class weights (CE expects [w_neg, w_pos]) ---
    n_total = train_n_pos + train_n_neg
    w_neg = n_total / (2.0 * max(train_n_neg, 1))
    w_pos = n_total / (2.0 * max(train_n_pos, 1))
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    criterion_bin = nn.CrossEntropyLoss(weight=class_weights)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # AMP: use autocast for all devices, but GradScaler only for CUDA
    use_amp = True  # Use autocast for all devices
    if device.type == 'cuda':
        scaler = GradScaler()
        print(f"Using AMP with autocast and GradScaler (CUDA device)")
    else:
        scaler = None
        print(f"Using AMP with autocast only (no GradScaler for {device.type} device)")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss_bin, train_loss_attr, train_loss, train_acc = train_epoch(
            model, train_loader, criterion_bin, optimizer, device, scaler, args.attr_weight, use_amp
        )
        print(f"  Train Loss (bin): {train_loss_bin:.4f}")
        print(f"  Train Loss (attr): {train_loss_attr:.4f}")
        print(f"  Train Loss (total): {train_loss:.4f}")
        print(f"  Train Acc (bin): {train_acc:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion_bin, device, attr_cols, use_amp)
        print(f"  Val Loss (bin): {val_metrics['loss_bin']:.4f}")
        print(f"  Val Loss (attr): {val_metrics['loss_attr']:.4f}")
        
        # Binary metrics
        if val_metrics['binary_pr_auc'] is not None:
            print(f"  Val Binary PR-AUC: {val_metrics['binary_pr_auc']:.4f}")
            print(f"  Val Binary ROC-AUC: {val_metrics['binary_roc_auc']:.4f}")
        else:
            print("  Val Binary PR-AUC: N/A")
            print("  Val Binary ROC-AUC: N/A")
        
        # Attribute metrics
        print("  Val Attribute PR-AUC:")
        for attr in attr_cols:
            pr_auc = val_metrics['attr_metrics'].get(f'{attr}_pr_auc')
            if pr_auc is not None:
                print(f"    {attr}: {pr_auc:.4f}")
            else:
                print(f"    {attr}: N/A")
        
        # mAP metrics
        if val_metrics['micro_map'] is not None:
            print(f"  Val Micro mAP: {val_metrics['micro_map']:.4f}")
        else:
            print("  Val Micro mAP: N/A")
        
        if val_metrics['macro_map'] is not None:
            print(f"  Val Macro mAP: {val_metrics['macro_map']:.4f}")
        else:
            print("  Val Macro mAP: N/A")
        
        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.2f}s")
        print()
    
    # Save model state_dict
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model state_dict saved to: {out_path.absolute()}")
    
    # Compute prevalences and save metadata JSON
    # Combine train and val for prevalence computation
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    prevalences = compute_prevalences(combined_df, attr_cols)
    metadata = {
        'attr_cols': attr_cols,
        'prevalences': prevalences
    }
    
    metadata_path = out_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path.absolute()}")


if __name__ == '__main__':
    main()

