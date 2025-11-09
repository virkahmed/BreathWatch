#!/usr/bin/env python3
"""
Train a small CNN to classify ICBHI tiles as wheeze vs other.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


class WheezeDataset(Dataset):
    """Dataset for wheeze classification."""
    
    def __init__(self, manifest_df: pd.DataFrame):
        """
        Args:
            manifest_df: DataFrame with columns: path_npy, label
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
    
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
        
        # Label: 1 for wheeze, 0 for other
        label = 1 if row['label'] == 'wheeze' else 0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return spec_tensor, label_tensor


class WheezeCNN(nn.Module):
    """Small CNN for wheeze classification."""
    
    def __init__(self):
        super(WheezeCNN, self).__init__()
        
        # Conv2d(1,16,3,pad=1) → ReLU → BN → MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        # Conv2d(16,32,3,pad=1) → ReLU → BN → MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Conv2d(32,64,3,pad=1) → ReLU → BN
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        
        # AdaptiveAvgPool2d(1) → Linear(64→2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)
    
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        # Global pooling and classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x


def get_device():
    """Get the best available device (MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for spec, label in tqdm(dataloader, desc="Training", leave=False):
        spec = spec.to(device)
        label = label.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(spec)
        loss = criterion(logits, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs_wheeze = []
    
    with torch.no_grad():
        for spec, label in tqdm(dataloader, desc="Validating", leave=False):
            spec = spec.to(device)
            label = label.to(device)
            
            # Forward pass
            logits = model(spec)
            loss = criterion(logits, label)
            
            # Statistics
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            p_wheeze = probs[:, 1].cpu().numpy()
            
            all_labels.extend(label.cpu().numpy())
            all_probs_wheeze.extend(p_wheeze)
    
    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_probs_wheeze = np.array(all_probs_wheeze)
    
    # Compute metrics
    metrics = {'loss': avg_loss}
    
    if SKLEARN_AVAILABLE:
        try:
            pr_auc = average_precision_score(all_labels, all_probs_wheeze)
            roc_auc = roc_auc_score(all_labels, all_probs_wheeze)
            metrics['pr_auc'] = pr_auc
            metrics['roc_auc'] = roc_auc
        except Exception as e:
            print(f"Warning: Error computing AUC metrics: {e}", file=sys.stderr)
            metrics['pr_auc'] = None
            metrics['roc_auc'] = None
    else:
        metrics['pr_auc'] = None
        metrics['roc_auc'] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train a CNN to classify ICBHI tiles as wheeze vs other'
    )
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to icbhi_wheeze_tiles.csv')
    parser.add_argument('--out', type=str, default='models/wheeze_head.pt',
                        help='Output model path (default: models/wheeze_head.pt)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of training epochs (default: 8)')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    
    args = parser.parse_args()
    
    # Check if scikit-learn is available
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn is not installed. AUC metrics will be skipped.", file=sys.stderr)
        print("To install: pip install scikit-learn", file=sys.stderr)
        print()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    manifest_df = pd.read_csv(args.manifest)
    
    # Filter to only wheeze and other labels
    manifest_df = manifest_df[manifest_df['label'].isin(['wheeze', 'other'])].copy()
    print(f"Found {len(manifest_df)} samples")
    print(f"  Wheeze: {len(manifest_df[manifest_df['label'] == 'wheeze'])}")
    print(f"  Other: {len(manifest_df[manifest_df['label'] == 'other'])}")
    
    # Split into train/val (85/15 stratified)
    if SKLEARN_AVAILABLE:
        train_df, val_df = train_test_split(
            manifest_df,
            test_size=0.15,
            stratify=manifest_df['label'],
            random_state=42
        )
    else:
        # Fallback: simple split without stratification
        print("Warning: Using non-stratified split (scikit-learn not available)", file=sys.stderr)
        train_size = int(0.85 * len(manifest_df))
        train_df = manifest_df.iloc[:train_size].copy()
        val_df = manifest_df.iloc[train_size:].copy()
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print()
    
    # Create datasets
    train_dataset = WheezeDataset(train_df)
    val_dataset = WheezeDataset(val_df)
    
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
    model = WheezeCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        if val_metrics['pr_auc'] is not None:
            print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        else:
            print("  Val PR-AUC: N/A (scikit-learn not available)")
            print("  Val ROC-AUC: N/A (scikit-learn not available)")
        
        print()
    
    # Save model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to: {out_path.absolute()}")


if __name__ == '__main__':
    main()

