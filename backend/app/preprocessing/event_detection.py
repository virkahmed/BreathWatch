"""
Event detection with hysteresis merging and attribute tagging.
Merges consecutive windows with cough detections into events and tags them.
"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from app.schemas import WindowPrediction, CoughEvent, CoughEventTag

logger = logging.getLogger(__name__)


def detect_cough_events_with_hysteresis(
    window_predictions: List[WindowPrediction],
    cough_threshold_start: float = 0.5,
    cough_threshold_end: float = 0.3,
    min_event_duration_ms: int = 100,
    max_gap_ms: int = 500
) -> List[CoughEvent]:
    """
    Detect cough events using hysteresis merging.
    
    Args:
        window_predictions: List of window predictions with p_cough and attributes
        cough_threshold_start: Threshold to start a new event (higher)
        cough_threshold_end: Threshold to end an event (lower, hysteresis)
        min_event_duration_ms: Minimum event duration in milliseconds
        max_gap_ms: Maximum gap between windows to merge into same event
    
    Returns:
        List of detected cough events
    """
    if not window_predictions:
        return []
    
    events = []
    current_event_start = None
    current_event_windows = []
    
    for i, window in enumerate(window_predictions):
        window_time_ms = i * 1000  # Each window is 1 second = 1000ms
        
        # Check if we should start a new event
        if window.p_cough >= cough_threshold_start:
            if current_event_start is None:
                # Start new event
                current_event_start = window_time_ms
                current_event_windows = [window]
            else:
                # Continue current event
                current_event_windows.append(window)
        elif window.p_cough >= cough_threshold_end:
            # Hysteresis: continue event if above lower threshold
            if current_event_start is not None:
                current_event_windows.append(window)
        else:
            # Below both thresholds, end event if we have one
            if current_event_start is not None:
                # Check gap from last window
                if current_event_windows:
                    last_window_time = (len(current_event_windows) - 1) * 1000
                    gap = window_time_ms - (current_event_start + last_window_time)
                    
                    if gap <= max_gap_ms:
                        # Small gap, continue event
                        current_event_windows.append(window)
                        continue
                
                # End the event
                event = _create_event_from_windows(
                    current_event_start,
                    window_time_ms,
                    current_event_windows,
                    min_event_duration_ms
                )
                if event:
                    events.append(event)
                
                current_event_start = None
                current_event_windows = []
    
    # Handle event that extends to end
    if current_event_start is not None and current_event_windows:
        end_time_ms = len(window_predictions) * 1000
        event = _create_event_from_windows(
            current_event_start,
            end_time_ms,
            current_event_windows,
            min_event_duration_ms
        )
        if event:
            events.append(event)
    
    logger.info(f"Detected {len(events)} cough events from {len(window_predictions)} windows")
    return events


def _create_event_from_windows(
    start_ms: int,
    end_ms: int,
    windows: List[WindowPrediction],
    min_duration_ms: int
) -> Optional[CoughEvent]:
    """Create a cough event from a list of windows."""
    duration_ms = end_ms - start_ms
    
    # Skip events that are too short
    if duration_ms < min_duration_ms:
        return None
    
    # Calculate average confidence
    confidences = [w.p_cough for w in windows]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Tag event based on attribute thresholds
    tags = _tag_event(windows)
    
    # Check quality (SNR)
    quality_flag = None
    snr_values = [w.snr for w in windows if w.snr is not None]
    if snr_values:
        avg_snr = sum(snr_values) / len(snr_values)
        if avg_snr < 5.0:  # Threshold for insufficient signal
            quality_flag = "INSUFFICIENT_SIGNAL"
    
    # Get window indices
    window_indices = [w.window_index for w in windows]
    
    return CoughEvent(
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=avg_confidence,
        tags=tags,
        quality_flag=quality_flag,
        window_indices=window_indices
    )


def _tag_event(windows: List[WindowPrediction]) -> List[CoughEventTag]:
    """
    Tag event based on attribute probabilities in windows.
    
    Tag if ≥40% of windows have attribute probability ≥ 0.5.
    """
    if not windows:
        return [CoughEventTag.COUGH_UNSPEC]
    
    num_windows = len(windows)
    threshold = 0.4  # 40% of windows
    min_windows = int(num_windows * threshold)
    
    tags = []
    
    # Count windows with each attribute above 0.5
    wet_count = sum(1 for w in windows if w.p_attr_wet >= 0.5)
    stridor_count = sum(1 for w in windows if w.p_attr_stridor >= 0.5)
    choking_count = sum(1 for w in windows if w.p_attr_choking >= 0.5)
    congestion_count = sum(1 for w in windows if w.p_attr_congestion >= 0.5)
    wheezing_count = sum(1 for w in windows if w.p_attr_wheezing_selfreport >= 0.5)
    
    if wet_count >= min_windows:
        tags.append(CoughEventTag.WET)
    if stridor_count >= min_windows:
        tags.append(CoughEventTag.STRIDOR)
    if choking_count >= min_windows:
        tags.append(CoughEventTag.CHOKING)
    if congestion_count >= min_windows:
        tags.append(CoughEventTag.CONGESTION)
    if wheezing_count >= min_windows:
        tags.append(CoughEventTag.SELFREPORTED_WHEEZING)
    
    # If no tags, use UNSPEC
    if not tags:
        tags.append(CoughEventTag.COUGH_UNSPEC)
    
    return tags


def calculate_snr(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Simple implementation: ratio of signal power to estimated noise floor.
    """
    try:
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise floor from quiet segments (bottom 10% of energy)
        energy = audio ** 2
        noise_floor = np.percentile(energy, 10)
        
        if noise_floor > 0:
            snr_db = 10 * np.log10(signal_power / noise_floor)
        else:
            snr_db = 0.0
        
        return max(0.0, snr_db)
    except Exception as e:
        logger.warning(f"Error calculating SNR: {e}")
        return 10.0  # Default reasonable SNR

