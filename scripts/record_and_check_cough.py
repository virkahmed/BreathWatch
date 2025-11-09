#!/usr/bin/env python3
"""
Quick script to sanity-check the cough_multitask model on a .wav file.

Usage (from repo root):
    python scripts/check_cough_on_wav.py

Requires:
    pip install librosa soundfile scipy
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import librosa
from scipy.ndimage import zoom

# --------------------------------------------------------------------------------------
# Paths & imports
# --------------------------------------------------------------------------------------

# This file lives in scripts/, so repo root is one level up
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from train_cough_multitask import CoughMultitaskCNN  # uses same arch as training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COUGH_MODEL_PATH = ROOT / "models" / "cough_multitask.pt"
WAV_PATH = ROOT / "test.wav"   # put your test audio here

# MUST be consistent with training / mk_spectros.py
TARGET_SR = 16000
N_MELS = 80
TILE_SECONDS = 1.0
TILE_STRIDE_SECONDS = 0.25    # 0.25s hop between tile starts
TARGET_SPEC_SHAPE = (256, 256)

# --------------------------------------------------------------------------------------
# Audio + spectrogram helpers
# --------------------------------------------------------------------------------------


def load_and_preprocess_audio(path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load mono audio, resample, and lightly trim silence."""
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)
    if y_trimmed.size == 0:
        y_trimmed = y
    return y_trimmed, target_sr


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    fmin: float = 20.0,
) -> np.ndarray:
    """Log-mel spectrogram in dB, aligned with typical cough model setup."""
    if n_fft is None:
        n_fft = int(0.025 * sr)       # 25 ms
    if hop_length is None:
        hop_length = int(0.010 * sr)  # 10 ms
    if win_length is None:
        win_length = n_fft

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, T)
    return log_mel


def slice_audio_tiles(
    audio: np.ndarray,
    sr: int,
    tile_seconds: float = TILE_SECONDS,
    stride_seconds: float = TILE_STRIDE_SECONDS,
) -> List[Tuple[int, int]]:
    """Return (start, end) sample indices for overlapping tiles."""
    tile_samples = int(tile_seconds * sr)
    stride_samples = int(stride_seconds * sr)

    if len(audio) < tile_samples:
        return []

    starts = range(0, len(audio) - tile_samples + 1, stride_samples)
    return [(s, s + tile_samples) for s in starts]


def resize_spectrogram(spec: np.ndarray, target_shape=TARGET_SPEC_SHAPE) -> np.ndarray:
    """Resize log-mel spectrogram to target_shape using bilinear-like interpolation."""
    h, w = spec.shape
    zh = target_shape[0] / h
    zw = target_shape[1] / w
    return zoom(spec, (zh, zw), order=1)


def make_tiles_from_wav(path: Path) -> torch.Tensor:
    """
    Full pipeline:
      - load + trim
      - tile in time
      - per-tile log-mel
      - resize to 256x256
      - normalize into [0,1]
    Returns:
      tiles: FloatTensor [N, 1, 256, 256]
    """
    audio, sr = load_and_preprocess_audio(path, TARGET_SR)
    tile_indices = slice_audio_tiles(audio, sr)

    if not tile_indices:
        raise ValueError("Audio too short to create any 1s tiles")

    specs = []
    for start, end in tile_indices:
        tile_audio = audio[start:end]
        spec = compute_log_mel_spectrogram(tile_audio, sr)      # [n_mels, T]
        spec = resize_spectrogram(spec, TARGET_SPEC_SHAPE)      # [256, 256]

        # Training-style normalization: shift/scale into [0,1]
        # Adjust if your training used a different rule.
        spec = (spec + 80.0) / 80.0
        spec = np.clip(spec, 0.0, 1.0)

        specs.append(spec.astype(np.float32))

    specs = np.stack(specs, axis=0)          # [N, 256, 256]
    specs = torch.from_numpy(specs)          # -> tensor
    specs = specs.unsqueeze(1)               # [N, 1, 256, 256]
    return specs


# --------------------------------------------------------------------------------------
# Model loader
# --------------------------------------------------------------------------------------


def load_cough_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")

    state_dict = torch.load(str(path), map_location=device)

    # Assumes training used CoughMultitaskCNN(num_attrs=5)
    # (wet, wheeze, stridor, choking, congestion)
    model = CoughMultitaskCNN(num_attrs=5)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Cough event grouping
# --------------------------------------------------------------------------------------


def group_cough_events(
    probs: np.ndarray,
    threshold: float,
    tile_seconds: float = TILE_SECONDS,
    stride_seconds: float = TILE_STRIDE_SECONDS,
) -> List[Tuple[float, float]]:
    """
    Convert per-tile probabilities into discrete cough events by merging
    consecutive tiles above threshold.

    Returns list of (start_time, end_time) in seconds.
    """
    events: List[Tuple[float, float]] = []
    current = None  # [start, end]

    for i, p in enumerate(probs):
        t_start = i * stride_seconds
        t_end = t_start + tile_seconds

        if p >= threshold:
            if current is None:
                current = [t_start, t_end]
            else:
                # extend event if overlapping/adjacent
                current[1] = max(current[1], t_end)
        else:
            if current is not None:
                events.append((current[0], current[1]))
                current = None

    if current is not None:
        events.append((current[0], current[1]))

    return events


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    if not WAV_PATH.exists():
        raise FileNotFoundError(
            f"Test audio not found at {WAV_PATH}. "
            f"Save your test clip as 'test.wav' in the repo root or update WAV_PATH."
        )

    print(f"Using audio:  {WAV_PATH}")
    print(f"Using model:  {COUGH_MODEL_PATH}")

    tiles = make_tiles_from_wav(WAV_PATH)   # [N, 1, 256, 256]
    print(f"Prepared {tiles.shape[0]} tiles")

    cough_model = load_cough_model(COUGH_MODEL_PATH, DEVICE)

    with torch.no_grad():
        out = cough_model(tiles.to(DEVICE))
        # CoughMultitaskCNN returns (logits_a, logits_b, ...)
        if isinstance(out, (list, tuple)):
            logits_a = out[0]
        else:
            logits_a = out

        if logits_a.ndim != 2 or logits_a.shape[1] < 2:
            raise RuntimeError(f"Unexpected logits shape from model: {logits_a.shape}")

        probs = torch.softmax(logits_a, dim=1)[:, 1].cpu().numpy()  # p(cough) per tile

    # Threshold & stats
    threshold = 0.5
    cough_tiles = np.where(probs >= threshold)[0]
    print(f"Tiles with p_cough >= {threshold}: {len(cough_tiles)}")

    # Top 10 tile scores
    if probs.size > 0:
        print("\nTop cough-like tiles:")
        top_idx = np.argsort(-probs)[:10]
        for i in top_idx:
            t_start = i * TILE_STRIDE_SECONDS
            print(f"  tile {i:4d}: t={t_start:6.2f}s  p_cough={probs[i]:.3f}")

    # Group into events
    events = group_cough_events(probs, threshold)
    print(f"\nEstimated cough events: {len(events)}")
    for start, end in events:
        print(f"  {start:6.2f}s – {end:6.2f}s")


if __name__ == "__main__":
    main()
