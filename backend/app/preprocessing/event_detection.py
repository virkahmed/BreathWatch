"""
Event detection with hysteresis merging and attribute tagging.
Merges consecutive windows with cough detections into events and tags them.

Includes both a simple reference-style event grouping and a more sophisticated
hysteresis-based approach.
"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from app.schemas import (
    WindowPrediction,
    CoughEvent,
    CoughEventTag,
    ProbabilityTimeline,
    EventSummary,
    ProbabilityEvent,
    AttributeVector,
    AttributeVectorSeries,
    AttributeFlags,
)

logger = logging.getLogger(__name__)


def group_cough_events_simple(
    window_predictions: List[WindowPrediction],
    threshold: float = 0.5,
    tile_seconds: float = 1.0,
    stride_seconds: float = 0.25,
    min_gap_ms: int = 3000,  # Minimum gap between events to count as separate coughs (3 seconds)
    min_drop: float = 0.15  # Minimum drop in p_cough to split events
) -> List[CoughEvent]:
    """
    Improved event grouping that better separates individual coughs.
    
    Converts per-tile probabilities into discrete cough events by merging
    consecutive tiles above threshold, but splits events when:
    - p_cough drops below threshold for at least min_gap_ms
    - p_cough drops significantly (by min_drop) even if still above threshold
    
    Args:
        window_predictions: List of window predictions with p_cough
        threshold: Probability threshold for cough detection (default 0.5)
        tile_seconds: Duration of each tile in seconds (1.0s)
        stride_seconds: Stride between tile starts in seconds (0.25s)
        min_gap_ms: Minimum gap in milliseconds between events to count as separate (default 3000ms = 3 seconds)
        min_drop: Minimum drop in p_cough to split events even if above threshold (default 0.15)
    
    Returns:
        List of detected cough events
    """
    if not window_predictions:
        return []
    
    events: List[CoughEvent] = []
    current_start = None
    current_windows = []
    last_high_window_idx = None  # Track last window with high p_cough
    peak_p_cough = 0.0  # Track peak p_cough in current event
    consecutive_low_count = 0  # Count consecutive windows below threshold
    last_high_time_ms = None  # Track time of last high window
    
    logger.info(f"🔍 Event detection: threshold={threshold}, min_gap={min_gap_ms}ms, min_drop={min_drop}")
    
    for i, window in enumerate(window_predictions):
        t_start_ms = i * stride_seconds * 1000  # Convert to milliseconds
        t_end_ms = t_start_ms + tile_seconds * 1000
        
        if window.p_cough >= threshold:
            # Above threshold
            # Check if we were below threshold BEFORE resetting the counter
            was_below_threshold = (consecutive_low_count > 0)
            consecutive_low_count = 0  # Reset low count
            
            if current_start is None:
                # Start new event
                logger.debug(f"  Window {i} (t={t_start_ms}ms): Starting NEW event (p_cough={window.p_cough:.3f})")
                current_start = t_start_ms
                current_windows = [window]
                peak_p_cough = window.p_cough
                last_high_window_idx = i
                last_high_time_ms = t_start_ms
            elif was_below_threshold:
                # We were below threshold and now back above - ALWAYS start a new event
                # This ensures that any gap (even 1 window = 0.25s) creates separate events
                if last_high_window_idx is not None:
                    last_high_end_ms = last_high_window_idx * stride_seconds * 1000 + tile_seconds * 1000
                    gap_ms = t_start_ms - last_high_end_ms
                    logger.info(f"  Window {i} (t={t_start_ms}ms): ⚠️ SPLITTING event - was below threshold, now back above (gap={gap_ms}ms, consecutive_low={consecutive_low_count} windows, p_cough={window.p_cough:.3f})")
                    # End previous event
                    event = _create_event_from_windows(
                        current_start,
                        last_high_end_ms,
                        current_windows,
                        min_duration_ms=100
                    )
                    if event:
                        events.append(event)
                        logger.info(f"  ✅ Created event #{len(events)}: {event.start_ms}ms-{event.end_ms}ms ({((event.end_ms - event.start_ms)/1000):.2f}s), confidence={event.confidence:.3f}")
                
                # Start new event
                logger.info(f"  Window {i} (t={t_start_ms}ms): Starting NEW event after gap (p_cough={window.p_cough:.3f})")
                current_start = t_start_ms
                current_windows = [window]
                peak_p_cough = window.p_cough
                last_high_window_idx = i
                last_high_time_ms = t_start_ms
            else:
                # Check if we should split the event
                # Split if there's a significant time gap since last high window (even if p_cough is still high)
                # This helps separate distinct coughs that might have similar p_cough values
                time_since_last_high = t_start_ms - last_high_time_ms if last_high_time_ms is not None else 0
                if time_since_last_high >= min_gap_ms:
                    # Significant time gap - split the event
                    logger.info(f"  Window {i} (t={t_start_ms}ms): ⚠️ SPLITTING event due to time gap ({time_since_last_high}ms >= {min_gap_ms}ms, p_cough={window.p_cough:.3f})")
                    if last_high_window_idx is not None:
                        prev_t_end_ms = last_high_window_idx * stride_seconds * 1000 + tile_seconds * 1000
                        event = _create_event_from_windows(
                            current_start,
                            prev_t_end_ms,
                            current_windows,
                            min_duration_ms=100
                        )
                        if event:
                            events.append(event)
                            logger.info(f"  ✅ Created event #{len(events)}: {event.start_ms}ms-{event.end_ms}ms ({((event.end_ms - event.start_ms)/1000):.2f}s), confidence={event.confidence:.3f}")
                    
                    # Start new event
                    current_start = t_start_ms
                    current_windows = [window]
                    peak_p_cough = window.p_cough
                    last_high_window_idx = i
                    last_high_time_ms = t_start_ms
                # Split if p_cough drops significantly from peak (even if still above threshold)
                elif peak_p_cough > 0 and (peak_p_cough - window.p_cough) >= min_drop:
                    # Significant drop - end current event and start new one
                    logger.info(f"  Window {i} (t={t_start_ms}ms): ⚠️ SPLITTING event due to p_cough drop ({window.p_cough:.3f} vs peak {peak_p_cough:.3f}, drop={peak_p_cough - window.p_cough:.3f})")
                    # End current event at the window before the drop
                    if last_high_window_idx is not None:
                        prev_t_end_ms = last_high_window_idx * stride_seconds * 1000 + tile_seconds * 1000
                        event = _create_event_from_windows(
                            current_start,
                            prev_t_end_ms,
                            current_windows,  # Already excludes the current window that caused the drop
                            min_duration_ms=100
                        )
                        if event:
                            events.append(event)
                            logger.info(f"  ✅ Created event #{len(events)}: {event.start_ms}ms-{event.end_ms}ms ({((event.end_ms - event.start_ms)/1000):.2f}s), confidence={event.confidence:.3f}")
                    
                    # Start new event with the current window
                    current_start = t_start_ms
                    current_windows = [window]
                    peak_p_cough = window.p_cough
                    last_high_window_idx = i
                    last_high_time_ms = t_start_ms
                else:
                    # Continue current event
                    current_windows.append(window)
                    if window.p_cough > peak_p_cough:
                        peak_p_cough = window.p_cough
                    last_high_window_idx = i
                    last_high_time_ms = t_start_ms
        else:
            # Below threshold
            consecutive_low_count += 1
            
            if current_start is not None:
                # We have an active event - check if we should end it
                if last_high_window_idx is not None:
                    # Calculate gap from end of last high window to start of current low window
                    last_high_end_ms = last_high_window_idx * stride_seconds * 1000 + tile_seconds * 1000
                    gap_ms = t_start_ms - last_high_end_ms
                    
                    # Calculate how many consecutive low windows we've had
                    low_window_duration_ms = consecutive_low_count * stride_seconds * 1000
                    
                    if gap_ms >= min_gap_ms:
                        # Long enough gap - end current event
                        logger.info(f"  Window {i} (t={t_start_ms}ms): ⚠️ ENDING event due to gap (gap={gap_ms}ms >= {min_gap_ms}ms, consecutive_low={consecutive_low_count} windows, p_cough={window.p_cough:.3f})")
                        event_end_ms = last_high_end_ms
                        event = _create_event_from_windows(
                            current_start,
                            event_end_ms,
                            current_windows,
                            min_duration_ms=100
                        )
                        if event:
                            events.append(event)
                            logger.info(f"  ✅ Created event #{len(events)}: {event.start_ms}ms-{event.end_ms}ms ({((event.end_ms - event.start_ms)/1000):.2f}s), confidence={event.confidence:.3f}")
                        
                        current_start = None
                        current_windows = []
                        peak_p_cough = 0.0
                        last_high_window_idx = None
                        last_high_time_ms = None
                        consecutive_low_count = 0
                    else:
                        # Not long enough gap yet - log for debugging
                        if consecutive_low_count == 1:  # Only log first low window
                            logger.debug(f"  Window {i} (t={t_start_ms}ms): Below threshold (p_cough={window.p_cough:.3f}), gap={gap_ms}ms < {min_gap_ms}ms, waiting...")
    
    # Handle event that extends to end
    if current_start is not None:
        # Last window ends at: (len-1) * stride + tile_seconds
        last_window_idx = len(window_predictions) - 1
        t_end_ms = last_window_idx * stride_seconds * 1000 + tile_seconds * 1000
        event = _create_event_from_windows(
            current_start,
            t_end_ms,
            current_windows,
            min_duration_ms=100
        )
        if event:
            events.append(event)
            logger.debug(f"  Created final event: {event.start_ms}ms-{event.end_ms}ms, confidence={event.confidence:.3f}")
    
    logger.info(f"✅ Detected {len(events)} cough events (improved grouping) from {len(window_predictions)} windows")
    if events:
        logger.info(f"   Event durations: {[f'{(e.end_ms - e.start_ms)/1000:.2f}s' for e in events]}")
    return events


def detect_cough_events_with_hysteresis(
    window_predictions: List[WindowPrediction],
    cough_threshold_start: float = 0.5,
    cough_threshold_end: float = 0.3,
    min_event_duration_ms: int = 100,
    max_gap_ms: int = 500,
    min_separation_ms: int = 200  # Minimum time between separate cough events
) -> List[CoughEvent]:
    """
    Detect cough events using hysteresis merging.
    
    Improved logic to better separate individual coughs:
    - Uses hysteresis to merge consecutive windows
    - Splits events if there's a significant drop in p_cough (even if above threshold)
    - Requires minimum separation between events to count as separate coughs
    
    Args:
        window_predictions: List of window predictions with p_cough and attributes
        cough_threshold_start: Threshold to start a new event (higher)
        cough_threshold_end: Threshold to end an event (lower, hysteresis)
        min_event_duration_ms: Minimum event duration in milliseconds
        max_gap_ms: Maximum gap between windows to merge into same event
        min_separation_ms: Minimum time between events to count as separate coughs
    
    Returns:
        List of detected cough events
    """
    if not window_predictions:
        return []
    
    events = []
    current_event_start = None
    current_event_windows = []
    last_event_end_ms = None
    low_cough_count = 0  # Count consecutive windows with low p_cough
    
    logger.debug(f"Detecting events from {len(window_predictions)} windows with thresholds: start={cough_threshold_start}, end={cough_threshold_end}")
    
    for i, window in enumerate(window_predictions):
        # Calculate window time from window_index
        # With stride=0.25s, windows are spaced 0.25s apart (not 1s)
        # window_index represents the sequential window number across all chunks
        # Each window is 1 second long, but starts every 0.25s
        # So window_index i starts at i * 0.25 seconds = i * 250ms
        # But we need to handle the case where window_index might be absolute across chunks
        # For now, use the index in the list (i) since we're processing one chunk at a time
        # The actual time will be adjusted later when combining chunks
        window_time_ms = i * 250  # 0.25s stride = 250ms per window index
        
        # Check if we should start a new event
        if window.p_cough >= cough_threshold_start:
            # Check if we should start a NEW event (not continue current one)
            if current_event_start is None:
                # Check if enough time has passed since last event
                if last_event_end_ms is None or (window_time_ms - last_event_end_ms) >= min_separation_ms:
                    # Start new event
                    current_event_start = window_time_ms
                    current_event_windows = [window]
                    low_cough_count = 0  # Reset counter
                    logger.debug(f"  Window {i}: Starting new event at {window_time_ms}ms (p_cough={window.p_cough:.3f})")
                else:
                    # Too soon after last event, might be continuation - but we'll treat as new if p_cough is high
                    # Actually, if we're here and current_event_start is None, we should start a new event
                    current_event_start = window_time_ms
                    current_event_windows = [window]
                    low_cough_count = 0  # Reset counter
                    logger.debug(f"  Window {i}: Starting new event (close to previous) at {window_time_ms}ms")
            else:
                # Continue current event
                current_event_windows.append(window)
                low_cough_count = 0  # Reset counter when continuing event
        elif window.p_cough >= cough_threshold_end:
            # Hysteresis: continue event if above lower threshold
            if current_event_start is not None:
                current_event_windows.append(window)
                low_cough_count = 0  # Reset low cough counter
        else:
            # Below lower threshold
            low_cough_count += 1
            
            # If we have multiple consecutive low-cough windows, end the event
            # This helps separate individual coughs that are close together
            if current_event_start is not None and low_cough_count >= 2:  # 2 windows = 0.5s at 0.25s stride
                logger.debug(f"  Window {i}: {low_cough_count} consecutive low-cough windows, ending event")
                # End the event at the start of the low-cough period
                event_end_ms = window_time_ms - (low_cough_count - 1) * 250
                event = _create_event_from_windows(
                    current_event_start,
                    event_end_ms,
                    current_event_windows[:-low_cough_count] if len(current_event_windows) > low_cough_count else current_event_windows,
                    min_event_duration_ms
                )
                if event:
                    events.append(event)
                    last_event_end_ms = event_end_ms
                    logger.debug(f"  Created event: {event.start_ms}ms-{event.end_ms}ms, confidence={event.confidence:.3f}")
                
                current_event_start = None
                current_event_windows = []
                low_cough_count = 0
                continue
            # Below both thresholds, end event if we have one
            if current_event_start is not None:
                # Check if there's a significant drop in p_cough that suggests end of cough
                # Even if we're in hysteresis zone, if p_cough drops significantly, end the event
                if len(current_event_windows) > 0:
                    # Get the peak p_cough in current event
                    peak_cough = max(w.p_cough for w in current_event_windows)
                    # If current window is significantly lower than peak, end event
                    if peak_cough - window.p_cough > 0.3:  # Significant drop
                        logger.debug(f"  Window {i}: Significant drop in p_cough ({window.p_cough:.3f} vs peak {peak_cough:.3f}), ending event")
                        # End the event
                        event = _create_event_from_windows(
                            current_event_start,
                            window_time_ms,
                            current_event_windows,
                            min_event_duration_ms
                        )
                        if event:
                            events.append(event)
                            last_event_end_ms = window_time_ms
                            logger.debug(f"  Created event: {event.start_ms}ms-{event.end_ms}ms, confidence={event.confidence:.3f}")
                        
                        current_event_start = None
                        current_event_windows = []
                        continue
                
                # Check gap from last window in event
                if current_event_windows:
                    # Find the index of the last window in the current event
                    # The last window in current_event_windows is at position len-1
                    # But we need its actual time position
                    # Since windows are in order, the last window's index in the full list is i-1
                    # Actually, we can calculate: current_event_start is the time of the first window
                    # The last window added is at position len(current_event_windows)-1
                    # If first window is at index start_idx, last is at start_idx + len-1
                    # Time of last window = (start_idx + len-1) * 250ms
                    # But we don't have start_idx, so use: current_event_start + (len-1)*250
                    last_window_in_event_time = current_event_start + (len(current_event_windows) - 1) * 250
                    gap = window_time_ms - last_window_in_event_time
                    
                    if gap <= max_gap_ms:
                        # Small gap, continue event (but we're below threshold, so this shouldn't happen often)
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
                    last_event_end_ms = window_time_ms
                    logger.debug(f"  Created event: {event.start_ms}ms-{event.end_ms}ms, confidence={event.confidence:.3f}")
                
                current_event_start = None
                current_event_windows = []
    
    # Handle event that extends to end
    if current_event_start is not None and current_event_windows:
        # Last window ends at: start_time + window_duration
        # With stride=0.25s, last window index is len-1, starts at (len-1)*250ms
        # Window is 1 second long, so ends at (len-1)*250 + 1000ms
        end_time_ms = (len(window_predictions) - 1) * 250 + 1000
        event = _create_event_from_windows(
            current_event_start,
            end_time_ms,
            current_event_windows,
            min_event_duration_ms
        )
        if event:
            events.append(event)
            logger.debug(f"  Created final event: {event.start_ms}ms-{event.end_ms}ms, confidence={event.confidence:.3f}")
    
    logger.info(f"✅ Detected {len(events)} cough events from {len(window_predictions)} windows")
    if events:
        logger.info(f"   Event durations: {[f'{(e.end_ms - e.start_ms)/1000:.2f}s' for e in events]}")
        logger.info(f"   Event confidences: {[f'{e.confidence:.3f}' for e in events]}")
    
    return events


def _create_event_from_windows(
    start_ms: int,
    end_ms: int,
    windows: List[WindowPrediction],
    min_duration_ms: int
) -> Optional[CoughEvent]:
    """
    Create a cough event from a list of windows.
    
    Each event represents a single cough detection.
    Multiple consecutive windows with high p_cough get merged into one event.
    """
    duration_ms = end_ms - start_ms
    
    # Skip events that are too short
    if duration_ms < min_duration_ms:
        logger.debug(f"  Skipping event: too short ({duration_ms}ms < {min_duration_ms}ms)")
        return None
    
    if not windows:
        logger.debug(f"  Skipping event: no windows")
        return None
    
    # Calculate average confidence
    confidences = [w.p_cough for w in windows]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0
    
    # Tag event based on attribute thresholds
    tags = _tag_event(windows)
    
    # Check quality (SNR)
    # Note: SNR filtering is currently disabled in main.py for increased sensitivity
    # Lowered threshold from 5.0 to 3.0 dB for less aggressive filtering if re-enabled
    quality_flag = None
    snr_values = [w.snr for w in windows if w.snr is not None]
    if snr_values:
        avg_snr = sum(snr_values) / len(snr_values)
        if avg_snr < 3.0:  # Lowered threshold from 5.0 to 3.0 dB for increased sensitivity
            quality_flag = "INSUFFICIENT_SIGNAL"
            logger.debug(f"  Event marked as INSUFFICIENT_SIGNAL (SNR={avg_snr:.2f}dB)")
    
    # Get window indices
    window_indices = [w.window_index for w in windows]
    
    logger.debug(f"  Creating event: {duration_ms}ms, {len(windows)} windows, confidence={avg_confidence:.3f} (max={max_confidence:.3f}), tags={tags}")
    
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
    wheezing_count = sum(1 for w in windows if w.p_attr_wheezing >= 0.5)
    
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


def build_probability_timeline(
    window_predictions: List[WindowPrediction],
    tile_seconds: float = 1.0,
    stride_seconds: float = 0.25
) -> ProbabilityTimeline:
    """
    Build dense probability timeline arrays for cough + attributes.
    """
    if not window_predictions:
        return ProbabilityTimeline(tile_seconds=tile_seconds, stride_seconds=stride_seconds)
    
    windows_sorted = sorted(window_predictions, key=lambda w: w.window_index)
    indices = [w.window_index for w in windows_sorted]
    times = [idx * stride_seconds for idx in indices]
    p_cough = [w.p_cough for w in windows_sorted]
    
    attr_series = AttributeVectorSeries(
        wet=[w.p_attr_wet for w in windows_sorted],
        wheezing=[w.p_attr_wheezing for w in windows_sorted],
        stridor=[w.p_attr_stridor for w in windows_sorted],
        choking=[w.p_attr_choking for w in windows_sorted],
        congestion=[w.p_attr_congestion for w in windows_sorted],
    )
    
    return ProbabilityTimeline(
        tile_seconds=tile_seconds,
        stride_seconds=stride_seconds,
        indices=indices,
        times=times,
        p_cough=p_cough,
        attr_series=attr_series,
    )


def summarize_probability_events(
    window_predictions: List[WindowPrediction],
    tile_seconds: float = 1.0,
    stride_seconds: float = 0.25,
    cough_threshold: float = 0.5,
    attr_flag_threshold: float = 0.7,
) -> EventSummary:
    """
    Merge contiguous cough-like tiles into events and compute attribute statistics.
    """
    if not window_predictions:
        return EventSummary()
    
    windows_sorted = sorted(window_predictions, key=lambda w: w.window_index)
    events: List[ProbabilityEvent] = []
    current_group: List[WindowPrediction] = []
    last_index: Optional[int] = None
    
    for window in windows_sorted:
        is_cough_like = window.p_cough >= cough_threshold
        if is_cough_like:
            if not current_group:
                current_group = [window]
            else:
                expected_next = current_group[-1].window_index + 1
                if window.window_index == expected_next:
                    current_group.append(window)
                else:
                    event = _finalize_probability_event(
                        current_group, tile_seconds, stride_seconds, attr_flag_threshold
                    )
                    if event:
                        events.append(event)
                    current_group = [window]
            last_index = window.window_index
        else:
            if current_group:
                event = _finalize_probability_event(
                    current_group, tile_seconds, stride_seconds, attr_flag_threshold
                )
                if event:
                    events.append(event)
                current_group = []
            last_index = window.window_index
    
    if current_group:
        event = _finalize_probability_event(
            current_group, tile_seconds, stride_seconds, attr_flag_threshold
        )
        if event:
            events.append(event)
    
    return EventSummary(num_events=len(events), events=events)


def _finalize_probability_event(
    windows: List[WindowPrediction],
    tile_seconds: float,
    stride_seconds: float,
    attr_flag_threshold: float,
) -> Optional[ProbabilityEvent]:
    """Convert a list of tiles into a ProbabilityEvent."""
    if not windows:
        return None
    
    tile_indices = [w.window_index for w in windows]
    start_time = tile_indices[0] * stride_seconds
    end_time = tile_indices[-1] * stride_seconds + tile_seconds
    duration = end_time - start_time
    
    p_values = [w.p_cough for w in windows]
    attr_probs = AttributeVector(
        wet=float(np.mean([w.p_attr_wet for w in windows])),
        wheezing=float(np.mean([w.p_attr_wheezing for w in windows])),
        stridor=float(np.mean([w.p_attr_stridor for w in windows])),
        choking=float(np.mean([w.p_attr_choking for w in windows])),
        congestion=float(np.mean([w.p_attr_congestion for w in windows])),
    )
    attr_flags = AttributeFlags(
        wet=int(attr_probs.wet >= attr_flag_threshold),
        wheezing=int(attr_probs.wheezing >= attr_flag_threshold),
        stridor=int(attr_probs.stridor >= attr_flag_threshold),
        choking=int(attr_probs.choking >= attr_flag_threshold),
        congestion=int(attr_probs.congestion >= attr_flag_threshold),
    )
    
    return ProbabilityEvent(
        start=float(start_time),
        end=float(end_time),
        duration=float(duration),
        tile_indices=tile_indices,
        p_cough_max=float(max(p_values)),
        p_cough_mean=float(sum(p_values) / len(p_values)),
        attr_probs=attr_probs,
        attr_flags=attr_flags,
    )
