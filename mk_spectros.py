#!/usr/bin/env python3
"""
Generate log-Mel spectrograms from CoughVid and ICBHI datasets.
Creates tiled spectrograms with associated manifest CSVs.
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def load_and_preprocess_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio, resample to target_sr, convert to mono, trim silence."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    # Lightly trim silence (top_db=40)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)
    return y_trimmed, target_sr


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = None,
    hop_length: int = None,
    win_length: int = None,
    fmin: float = 20.0
) -> np.ndarray:
    """
    Compute log-Mel spectrogram.
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands (80)
        n_fft: FFT window size (25 ms = 400 samples at 16 kHz)
        hop_length: Hop length (10 ms = 160 samples at 16 kHz)
        win_length: Window length (25 ms = 400 samples at 16 kHz)
        fmin: Minimum frequency (20 Hz)
    """
    if n_fft is None:
        n_fft = int(0.025 * sr)  # 25 ms window
    if hop_length is None:
        hop_length = int(0.010 * sr)  # 10 ms hop
    if win_length is None:
        win_length = n_fft
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def slice_audio_tiles(
    audio: np.ndarray,
    sr: int,
    tile_duration: float = 1.0,
    stride_duration: float = 0.25
) -> List[Tuple[int, int]]:
    """
    Generate tile boundaries for audio slicing.
    Returns list of (start_sample, end_sample) tuples.
    """
    tile_samples = int(tile_duration * sr)
    stride_samples = int(stride_duration * sr)
    
    tiles = []
    start = 0
    while start + tile_samples <= len(audio):
        end = start + tile_samples
        tiles.append((start, end))
        start += stride_samples
    
    return tiles


def resize_spectrogram(spec: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize spectrogram to target shape using interpolation."""
    zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
    resized = zoom(spec, zoom_factors, order=1)
    return resized


def save_spectrogram_tile(
    spec: np.ndarray,
    npy_path: str,
    png_path: str,
    target_shape: Tuple[int, int] = (256, 256)
):
    """Save spectrogram as .npy (float32) and .png (grayscale, no axes)."""
    # Resize to target shape
    spec_resized = resize_spectrogram(spec, target_shape)
    
    # Save as .npy (float32)
    np.save(npy_path, spec_resized.astype(np.float32))
    
    # Save as .png (grayscale, no axes)
    fig, ax = plt.subplots(figsize=(target_shape[1]/100, target_shape[0]/100), dpi=100)
    ax.imshow(spec_resized, aspect='auto', origin='lower', cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


def parse_icbhi_annotations(txt_path: str) -> List[Tuple[float, float, int, int]]:
    """
    Parse ICBHI annotation file.
    Returns list of (start_time, end_time, crackle, wheeze) tuples.
    """
    cycles = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                start_time = float(parts[0])
                end_time = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
                cycles.append((start_time, end_time, crackle, wheeze))
    return cycles


def tile_overlaps_wheeze_cycle(
    tile_start_sample: int,
    tile_end_sample: int,
    sr: int,
    wheeze_cycles: List[Tuple[float, float]]
) -> bool:
    """
    Check if tile overlaps ≥50% with any wheeze cycle.
    """
    tile_start_time = tile_start_sample / sr
    tile_end_time = tile_end_sample / sr
    tile_duration = tile_end_time - tile_start_time
    
    for cycle_start, cycle_end in wheeze_cycles:
        # Calculate overlap
        overlap_start = max(tile_start_time, cycle_start)
        overlap_end = min(tile_end_time, cycle_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        # Check if overlap is ≥50% of tile duration
        if overlap_duration >= 0.5 * tile_duration:
            return True
    
    return False


def process_coughvid_dataset(
    coughvid_root: str,
    out_dir: str,
    max_files: Optional[int] = None
) -> pd.DataFrame:
    """Process CoughVid dataset and create manifest."""
    coughvid_path = Path(coughvid_root)
    wav_files = list(coughvid_path.rglob('*.wav'))
    
    if max_files:
        wav_files = wav_files[:max_files]
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    tile_id = 0
    
    for wav_file in tqdm(wav_files, desc="Processing CoughVid"):
        try:
            # Load and preprocess
            audio, sr = load_and_preprocess_audio(str(wav_file))
            
            # Generate tiles
            tiles = slice_audio_tiles(audio, sr)
            
            for start_sample, end_sample in tiles:
                # Extract tile
                tile_audio = audio[start_sample:end_sample]
                
                # Compute spectrogram
                spec = compute_log_mel_spectrogram(tile_audio, sr)
                
                # Generate paths
                tile_id_str = f"cv_{tile_id:06d}"
                npy_path = out_path / f"{tile_id_str}.npy"
                png_path = out_path / f"{tile_id_str}.png"
                
                # Save spectrogram
                save_spectrogram_tile(spec, str(npy_path), str(png_path))
                
                # Add to manifest
                manifest_rows.append({
                    'id': tile_id_str,
                    'dataset': 'coughvid',
                    'label': 'cough',
                    'path_png': str(png_path),
                    'path_npy': str(npy_path),
                    'src': str(wav_file),
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'sr': sr
                })
                
                tile_id += 1
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}", file=sys.stderr)
            continue
    
    return pd.DataFrame(manifest_rows)


def process_icbhi_dataset(
    icbhi_root: str,
    out_dir: str,
    max_files: Optional[int] = None,
    make_cough_negatives: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process ICBHI dataset.
    Returns (wheeze_manifest_df, negatives_manifest_df).
    """
    icbhi_path = Path(icbhi_root)
    wav_files = list(icbhi_path.rglob('*.wav'))
    
    if max_files:
        wav_files = wav_files[:max_files]
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    wheeze_manifest_rows = []
    negatives_manifest_rows = []
    wheeze_tile_id = 0
    neg_tile_id = 0
    
    for wav_file in tqdm(wav_files, desc="Processing ICBHI"):
        try:
            # Find corresponding annotation file
            txt_file = wav_file.with_suffix('.txt')
            if not txt_file.exists():
                # Try alternative naming (e.g., same name in parent dir)
                txt_file = wav_file.parent / f"{wav_file.stem}.txt"
                if not txt_file.exists():
                    continue
            
            # Parse annotations
            cycles = parse_icbhi_annotations(str(txt_file))
            
            # Separate wheeze cycles and negative cycles
            wheeze_cycles = [(s, e) for s, e, c, w in cycles if w == 1]
            negative_cycles = [(s, e) for s, e, c, w in cycles if c == 0 and w == 0]
            
            # Load and preprocess
            audio, sr = load_and_preprocess_audio(str(wav_file))
            
            # Process for wheeze classification
            tiles = slice_audio_tiles(audio, sr)
            
            for start_sample, end_sample in tiles:
                # Check if tile overlaps with wheeze cycle
                is_wheeze = tile_overlaps_wheeze_cycle(
                    start_sample, end_sample, sr, wheeze_cycles
                )
                
                label = 'wheeze' if is_wheeze else 'other'
                
                # Extract tile
                tile_audio = audio[start_sample:end_sample]
                
                # Compute spectrogram
                spec = compute_log_mel_spectrogram(tile_audio, sr)
                
                # Generate paths
                tile_id_str = f"icbhi_wheeze_{wheeze_tile_id:06d}"
                npy_path = out_path / f"{tile_id_str}.npy"
                png_path = out_path / f"{tile_id_str}.png"
                
                # Save spectrogram
                save_spectrogram_tile(spec, str(npy_path), str(png_path))
                
                # Add to manifest
                wheeze_manifest_rows.append({
                    'id': tile_id_str,
                    'dataset': 'icbhi',
                    'label': label,
                    'path_png': str(png_path),
                    'path_npy': str(npy_path),
                    'src': str(wav_file),
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'sr': sr
                })
                
                wheeze_tile_id += 1
            
            # Process for cough negatives (if requested)
            if make_cough_negatives and negative_cycles:
                # Sample one 1s slice from negative cycles
                cycle = random.choice(negative_cycles)
                cycle_start_time, cycle_end_time = cycle
                cycle_duration = cycle_end_time - cycle_start_time
                
                if cycle_duration >= 1.0:
                    # Sample a 1s window within the cycle
                    max_start = cycle_start_time
                    max_end = cycle_end_time - 1.0
                    if max_end >= max_start:
                        start_time = random.uniform(max_start, max_end)
                        end_time = start_time + 1.0
                        
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        
                        if end_sample <= len(audio):
                            # Extract tile
                            tile_audio = audio[start_sample:end_sample]
                            
                            # Compute spectrogram
                            spec = compute_log_mel_spectrogram(tile_audio, sr)
                            
                            # Generate paths
                            tile_id_str = f"icbhi_neg_{neg_tile_id:06d}"
                            npy_path = out_path / f"{tile_id_str}.npy"
                            png_path = out_path / f"{tile_id_str}.png"
                            
                            # Save spectrogram
                            save_spectrogram_tile(spec, str(npy_path), str(png_path))
                            
                            # Add to manifest
                            negatives_manifest_rows.append({
                                'id': tile_id_str,
                                'dataset': 'icbhi',
                                'label': 'no_cough',
                                'path_png': str(png_path),
                                'path_npy': str(npy_path),
                                'src': str(wav_file),
                                'start_sample': start_sample,
                                'end_sample': end_sample,
                                'sr': sr
                            })
                            
                            neg_tile_id += 1
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}", file=sys.stderr)
            continue
    
    wheeze_df = pd.DataFrame(wheeze_manifest_rows)
    negatives_df = pd.DataFrame(negatives_manifest_rows)
    
    return wheeze_df, negatives_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate log-Mel spectrograms from CoughVid and ICBHI datasets'
    )
    parser.add_argument('--coughvid-root', type=str, required=True,
                        help='Path to CoughVid dataset root')
    parser.add_argument('--icbhi-root', type=str, required=True,
                        help='Path to ICBHI dataset root')
    parser.add_argument('--out', type=str, default='data/feats',
                        help='Output directory (default: data/feats)')
    parser.add_argument('--make-cough-negatives', action='store_true',
                        help='Generate negative samples from ICBHI for cough classification')
    parser.add_argument('--max-cv-files', type=int, default=None,
                        help='Maximum number of CoughVid files to process')
    parser.add_argument('--max-icbhi-files', type=int, default=None,
                        help='Maximum number of ICBHI files to process')
    
    args = parser.parse_args()
    
    # Create output directory structure
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_path / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Process CoughVid
    print("Processing CoughVid dataset...")
    coughvid_df = process_coughvid_dataset(
        args.coughvid_root,
        str(out_path),
        args.max_cv_files
    )
    
    # Process ICBHI
    print("Processing ICBHI dataset...")
    icbhi_wheeze_df, icbhi_negatives_df = process_icbhi_dataset(
        args.icbhi_root,
        str(out_path),
        args.max_icbhi_files,
        args.make_cough_negatives
    )
    
    # Save manifests
    coughvid_csv = manifest_dir / 'coughvid_tiles.csv'
    icbhi_csv = manifest_dir / 'icbhi_wheeze_tiles.csv'
    
    coughvid_df.to_csv(coughvid_csv, index=False)
    icbhi_wheeze_df.to_csv(icbhi_csv, index=False)
    
    # If negatives were created, append to coughvid manifest or save separately
    if args.make_cough_negatives and not icbhi_negatives_df.empty:
        # Append negatives to coughvid manifest
        combined_df = pd.concat([coughvid_df, icbhi_negatives_df], ignore_index=True)
        combined_df.to_csv(coughvid_csv, index=False)
    
    # Print full paths
    print(f"\n✓ Successfully created manifests:")
    print(f"  {coughvid_csv.absolute()}")
    print(f"  {icbhi_csv.absolute()}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  CoughVid tiles: {len(coughvid_df)}")
    print(f"  ICBHI wheeze tiles: {len(icbhi_wheeze_df)}")
    if args.make_cough_negatives:
        print(f"  ICBHI negative tiles: {len(icbhi_negatives_df)}")


if __name__ == '__main__':
    main()

