"""
Mobile-optimized feature extraction for 1-second log-Mel windows.
Matches the mobile pipeline: 20-30ms frames → 1s windows.
"""
import numpy as np
import librosa
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_1s_logmel_window(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract 1-second log-Mel spectrogram window.
    
    This matches the mobile pipeline:
    - 20-30ms frames (handled by hop_length)
    - 1-second windows
    - Log-Mel spectrogram
    
    Args:
        audio: Audio array (should be ~1 second at sr)
        sr: Sample rate (16kHz)
        window_length: Window length in seconds (1.0s)
        n_mels: Number of mel filter banks (128)
        n_fft: FFT window size (2048 = ~128ms at 16kHz)
        hop_length: Hop length (512 = ~32ms at 16kHz, matches 20-30ms frames)
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sr/2)
    
    Returns:
        Log-Mel spectrogram array (n_mels, time_frames)
        For 1s at 16kHz with hop=512: ~31 time frames
    """
    # Ensure audio is at least window_length
    min_samples = int(window_length * sr)
    if len(audio) < min_samples:
        # Pad with zeros
        audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
    elif len(audio) > min_samples:
        # Truncate to window_length
        audio = audio[:min_samples]
    
    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (log-Mel)
    # Use log1p to avoid log(0), then normalize
    mel_spec_log = np.log1p(mel_spec)
    
    # Normalize to [0, 1] range for model input
    mel_spec_min = np.min(mel_spec_log)
    mel_spec_max = np.max(mel_spec_log)
    if mel_spec_max > mel_spec_min:
        mel_spec_log = (mel_spec_log - mel_spec_min) / (mel_spec_max - mel_spec_min)
    
    logger.debug(f"Extracted 1s log-Mel window: shape {mel_spec_log.shape}")
    return mel_spec_log


def segment_audio_1s_windows(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    overlap: float = 0.0
) -> list[Tuple[np.ndarray, float]]:
    """
    Segment audio into 1-second windows with optional overlap.
    
    Args:
        audio: Audio array
        sr: Sample rate
        window_length: Window length in seconds (1.0s)
        overlap: Overlap between windows in seconds (0.0 = no overlap)
    
    Returns:
        List of tuples (window_audio, start_time)
    """
    window_samples = int(window_length * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = window_samples - overlap_samples
    
    windows = []
    start_idx = 0
    
    while start_idx + window_samples <= len(audio):
        window = audio[start_idx:start_idx + window_samples]
        start_time = start_idx / sr
        windows.append((window, start_time))
        start_idx += hop_samples
    
    # Add final window if there's remaining audio (pad if needed)
    if start_idx < len(audio):
        window = audio[start_idx:]
        if len(window) >= window_samples // 2:  # Only if at least half a window
            # Pad to full window
            window = np.pad(window, (0, window_samples - len(window)), mode='constant')
            windows.append((window, start_idx / sr))
    
    return windows


def prepare_mobile_features(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    n_mels: int = 128
) -> np.ndarray:
    """
    Prepare features for mobile model input.
    
    Args:
        audio: Audio array (1 second)
        sr: Sample rate
        window_length: Window length (1.0s)
        n_mels: Number of mel bins
    
    Returns:
        Feature array ready for model: (1, n_mels, time_frames)
    """
    # Extract 1s log-Mel window
    mel_spec = extract_1s_logmel_window(
        audio, sr=sr, window_length=window_length, n_mels=n_mels
    )
    
    # Add channel dimension for CNN
    if len(mel_spec.shape) == 2:
        mel_spec = np.expand_dims(mel_spec, axis=0)  # (1, height, width)
    
    return mel_spec

