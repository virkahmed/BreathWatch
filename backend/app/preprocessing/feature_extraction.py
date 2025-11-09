"""
Feature extraction utilities for ML models.
"""
import logging
import numpy as np
import librosa
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract Mel spectrogram from audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mels: Number of mel filter banks
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sr/2)
    
    Returns:
        Mel spectrogram array (n_mels, time_frames)
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        logger.debug(f"Extracted Mel spectrogram: shape {mel_spec_db.shape}")
        return mel_spec_db
    except Exception as e:
        logger.error(f"Error extracting Mel spectrogram: {e}")
        raise


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel filter banks
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        MFCC array (n_mfcc, time_frames)
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        logger.debug(f"Extracted MFCC: shape {mfcc.shape}")
        return mfcc
    except Exception as e:
        logger.error(f"Error extracting MFCC: {e}")
        raise


def prepare_features_for_model(
    audio: np.ndarray,
    sr: int = 16000,
    feature_type: str = "mel_spectrogram",
    target_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Prepare features for ML model input.
    
    Args:
        audio: Audio array
        sr: Sample rate
        feature_type: Type of features ("mel_spectrogram" or "mfcc")
        target_shape: Target shape for model (height, width). If None, uses original shape.
    
    Returns:
        Feature array ready for model input
    """
    if feature_type == "mel_spectrogram":
        features = extract_mel_spectrogram(audio, sr)
    elif feature_type == "mfcc":
        features = extract_mfcc(audio, sr)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # Resize if target shape is specified
    if target_shape is not None:
        from scipy.ndimage import zoom
        current_shape = features.shape
        zoom_factors = (
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1]
        )
        features = zoom(features, zoom_factors, order=1)
        logger.debug(f"Resized features to {target_shape}")
    
    # Add channel dimension if needed (for CNN models)
    if len(features.shape) == 2:
        features = np.expand_dims(features, axis=0)  # (1, height, width)
    
    # Normalize to [0, 1] range
    features_min = np.min(features)
    features_max = np.max(features)
    if features_max > features_min:
        features = (features - features_min) / (features_max - features_min)
    
    logger.debug(f"Prepared features: shape {features.shape}, dtype {features.dtype}")
    return features

