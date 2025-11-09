"""
Pydantic models for request/response validation.
Pydantic v2 compatible with FastAPI.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis endpoint."""
    patient_id: Optional[int] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(M|F|Other)$", description="Patient sex")
    sleep_duration_minutes: Optional[float] = Field(None, ge=0, description="Sleep duration in minutes")


class DetectedEvent(BaseModel):
    """Model for a detected respiratory event."""
    event_type: str = Field(..., description="Type of event: cough, wheeze")
    timestamp: float = Field(..., description="Timestamp in seconds from start")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence probability")
    window_index: int = Field(..., description="Index of the 1-second window")


class DedalusInterpretation(BaseModel):
    """Model for Dedalus AI interpretation."""
    interpretation: str = Field(..., description="Health interpretation text")
    severity: Optional[str] = Field(None, description="Severity level: mild, moderate, severe")
    recommendations: Optional[List[str]] = Field(None, description="Health recommendations")


class MobileSummaryRequest(BaseModel):
    """Summary from mobile app after local processing."""
    patient_id: Optional[int] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(M|F|Other)$", description="Patient sex")
    sleep_duration_minutes: Optional[float] = Field(None, ge=0, description="Sleep duration in minutes")
    cough_count: int = Field(..., ge=0, description="Total number of coughs detected")
    wheeze_count: int = Field(..., ge=0, description="Total number of wheeze events detected")
    cough_probability_avg: float = Field(..., ge=0.0, le=1.0, description="Average cough probability")
    wheeze_probability_avg: float = Field(..., ge=0.0, le=1.0, description="Average wheeze probability")
    detected_events: List[DetectedEvent] = Field(..., description="All detected events with timestamps")


class NightlySummaryResponse(BaseModel):
    """Response model for nightly summary endpoint."""
    dedalus_interpretation: Optional[DedalusInterpretation] = Field(None, description="Dedalus AI interpretation")
    summary_id: Optional[str] = Field(None, description="Optional summary identifier for storage")


class WindowPrediction(BaseModel):
    """Per 1-second window predictions."""
    window_index: int = Field(..., description="Index of the 1-second window")
    p_cough: float = Field(..., ge=0.0, le=1.0, description="Cough probability")
    p_attr_wet: float = Field(..., ge=0.0, le=1.0, description="Wet cough attribute probability")
    p_attr_wheezing: float = Field(..., ge=0.0, le=1.0, description="Wheezing attribute probability")
    p_attr_stridor: float = Field(..., ge=0.0, le=1.0, description="Stridor attribute probability")
    p_attr_choking: float = Field(..., ge=0.0, le=1.0, description="Choking attribute probability")
    p_attr_congestion: float = Field(..., ge=0.0, le=1.0, description="Congestion attribute probability")
    p_wheeze: float = Field(..., ge=0.0, le=1.0, description="Wheeze probability (from wheeze model)")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio for quality assessment")


class CoughEventTag(str, Enum):
    """Cough event tags."""
    WET = "WET"
    STRIDOR = "STRIDOR"
    CHOKING = "CHOKING"
    CONGESTION = "CONGESTION"
    SELFREPORTED_WHEEZING = "SELFREPORTED_WHEEZING"
    COUGH_UNSPEC = "COUGH_UNSPEC"


class CoughEvent(BaseModel):
    """Detected cough event with tags and attributes."""
    start_ms: int = Field(..., description="Event start time in milliseconds")
    end_ms: int = Field(..., description="Event end time in milliseconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Event confidence")
    tags: List[CoughEventTag] = Field(default_factory=list, description="Event tags")
    quality_flag: Optional[str] = Field(None, description="Quality flag (e.g., INSUFFICIENT_SIGNAL)")
    window_indices: List[int] = Field(default_factory=list, description="Window indices included in event")


class AttributeVector(BaseModel):
    """Probability vector for cough attributes."""
    wet: float = Field(0.0, ge=0.0, le=1.0)
    wheezing: float = Field(0.0, ge=0.0, le=1.0)
    stridor: float = Field(0.0, ge=0.0, le=1.0)
    choking: float = Field(0.0, ge=0.0, le=1.0)
    congestion: float = Field(0.0, ge=0.0, le=1.0)


class AttributeVectorSeries(BaseModel):
    """Timeline series for attribute probabilities."""
    wet: List[float] = Field(default_factory=list)
    wheezing: List[float] = Field(default_factory=list)
    stridor: List[float] = Field(default_factory=list)
    choking: List[float] = Field(default_factory=list)
    congestion: List[float] = Field(default_factory=list)


class AttributeFlags(BaseModel):
    """Binary attribute flags derived from probability thresholds."""
    wet: int = Field(0, ge=0, le=1)
    wheezing: int = Field(0, ge=0, le=1)
    stridor: int = Field(0, ge=0, le=1)
    choking: int = Field(0, ge=0, le=1)
    congestion: int = Field(0, ge=0, le=1)


class ProbabilityEvent(BaseModel):
    """Probability-aware cough event derived from contiguous tiles."""
    start: float = Field(..., ge=0.0, description="Event start time in seconds from recording start")
    end: float = Field(..., ge=0.0, description="Event end time in seconds from recording start")
    duration: float = Field(..., ge=0.0, description="Event duration in seconds")
    tile_indices: List[int] = Field(default_factory=list, description="Tile indices included in the event")
    p_cough_max: float = Field(..., ge=0.0, le=1.0, description="Maximum cough probability inside event")
    p_cough_mean: float = Field(..., ge=0.0, le=1.0, description="Mean cough probability inside event")
    attr_probs: AttributeVector = Field(default_factory=AttributeVector, description="Mean attribute probabilities for the event")
    attr_flags: AttributeFlags = Field(default_factory=AttributeFlags, description="Binary attribute flags using strict thresholds (e.g., >=0.7)")


class EventSummary(BaseModel):
    """Collection of cough probability events."""
    num_events: int = Field(0, ge=0, description="Number of events")
    events: List[ProbabilityEvent] = Field(default_factory=list, description="Detailed cough events")


class ProbabilityTimeline(BaseModel):
    """Timeline of per-tile probabilities for cough and attributes."""
    tile_seconds: float = Field(1.0, gt=0.0, description="Length of each tile in seconds")
    stride_seconds: float = Field(0.25, gt=0.0, description="Stride between tile starts in seconds")
    indices: List[int] = Field(default_factory=list, description="Tile indices (absolute across session)")
    times: List[float] = Field(default_factory=list, description="Tile start times (seconds from session start)")
    p_cough: List[float] = Field(default_factory=list, description="Cough probability per tile")
    attr_series: AttributeVectorSeries = Field(default_factory=AttributeVectorSeries, description="Attribute probability series")


class SymptomForm(BaseModel):
    """User symptom inputs for pattern panel."""
    fever: bool = Field(False, description="Fever present")
    sore_throat: bool = Field(False, description="Sore throat present")
    chest_tightness: bool = Field(False, description="Chest tightness present")
    duration: int = Field(0, ge=0, description="Duration in days")
    nocturnal_worsening: bool = Field(False, description="Symptoms worsen at night")
    asthma_history: bool = Field(False, description="History of asthma")
    copd_history: bool = Field(False, description="History of COPD")
    age_band: Optional[str] = Field(None, description="Age band (e.g., '18-30', '31-50', '51+')")
    smoker: bool = Field(False, description="Smoking status")


class PatternScore(BaseModel):
    """Pattern panel score with uncertainty."""
    pattern_name: str = Field(..., description="Pattern name (e.g., 'Asthma-like', 'COPD-like')")
    score: float = Field(..., ge=0.0, le=1.0, description="Pattern score")
    uncertainty_lower: float = Field(..., ge=0.0, le=1.0, description="Lower bound of uncertainty")
    uncertainty_upper: float = Field(..., ge=0.0, le=1.0, description="Upper bound of uncertainty")
    why: str = Field(..., description="Explanation for the pattern score")


class AttributePrevalence(BaseModel):
    """Attribute prevalence statistics."""
    wet: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of wet-like tiles/events")
    wheezing: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of wheezing-like tiles/events")
    stridor: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of stridor-like tiles/events")
    choking: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of choking-like tiles/events")
    congestion: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of congestion-like tiles/events")


class HourlyMetrics(BaseModel):
    """Metrics for a single hour of sleep."""
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    cough_count: int = Field(0, description="Number of coughs in this hour")
    wheeze_percent: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of time with wheeze in this hour")
    events: List[CoughEvent] = Field(default_factory=list, description="Cough events in this hour")


class TrendComparison(BaseModel):
    """Trend comparison data vs historical baselines."""
    vs_last_night: Dict[str, float] = Field(default_factory=dict, description="Comparison vs last night (change values)")
    vs_7_day_avg: Dict[str, float] = Field(default_factory=dict, description="Comparison vs 7-day average (change values)")
    percent_changes: Dict[str, float] = Field(default_factory=dict, description="Percent changes for key metrics")


class QualityMetrics(BaseModel):
    """Audio quality and reliability metrics."""
    avg_snr: float = Field(..., description="Average signal-to-noise ratio")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    low_quality_periods_count: int = Field(0, description="Number of periods with insufficient signal")
    high_confidence_events_count: int = Field(0, description="Number of events with confidence > 0.8")
    suppressed_events_count: int = Field(0, description="Number of events suppressed due to low quality")


class DisplayStrings(BaseModel):
    """Formatted strings for UI display."""
    sleep_duration_formatted: str = Field(..., description="Formatted sleep duration (e.g., '7h 30m')")
    coughs_per_hour_formatted: str = Field(..., description="Formatted coughs per hour (e.g., '2.6 /hr')")
    severity_badge_color: str = Field(..., description="Severity badge color: 'green', 'yellow', 'red'")
    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall health quality score (0-100)")


class NightlySummary(BaseModel):
    """Complete nightly summary with all metrics."""
    session_id: str = Field(..., description="Session identifier")
    patient_id: Optional[int] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    total_duration_minutes: float = Field(..., description="Total sleep duration in minutes")
    
    # Cough metrics
    coughs_per_hour: float = Field(0.0, description="Coughs per hour")
    bout_count: int = Field(0, description="Number of cough bouts")
    bout_lengths: List[float] = Field(default_factory=list, description="Length of each bout in seconds")
    inter_cough_intervals: List[float] = Field(default_factory=list, description="Time between cough events in seconds")
    avg_bout_length_seconds: Optional[float] = Field(None, description="Average bout length in seconds")
    avg_inter_cough_interval_seconds: Optional[float] = Field(None, description="Average inter-cough interval in seconds")
    max_coughs_in_single_hour: int = Field(0, description="Maximum coughs in any single hour")
    
    # Wheeze metrics
    wheeze_time_percent: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of time with wheeze")
    longest_wheeze_duration_seconds: Optional[float] = Field(None, description="Longest continuous wheeze period")
    wheeze_intensity_avg: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average wheeze probability during wheeze periods")
    
    # Attribute prevalence
    attribute_prevalence: AttributePrevalence = Field(..., description="Attribute prevalence statistics")
    
    # Events
    cough_events: List[CoughEvent] = Field(default_factory=list, description="All detected cough events (legacy representation)")
    event_summary: EventSummary = Field(default_factory=EventSummary, description="Probability-based cough events")
    probability_timeline: ProbabilityTimeline = Field(default_factory=ProbabilityTimeline, description="Full-recording probability trajectory")
    
    # Pattern panel
    pattern_scores: List[PatternScore] = Field(default_factory=list, description="Pattern panel scores")
    
    # Trend
    trend_arrow: Optional[str] = Field(None, description="Trend vs baseline: ↑ / ↔ / ↓")
    
    # UI-specific fields
    hourly_breakdown: List[HourlyMetrics] = Field(default_factory=list, description="Hourly breakdown of metrics")
    trend_comparison: Optional[TrendComparison] = Field(None, description="Trend comparisons vs historical data")
    quality_metrics: QualityMetrics = Field(..., description="Audio quality and reliability metrics")
    display_strings: DisplayStrings = Field(..., description="Formatted strings for UI display")
    
    # Dedalus interpretation (optional)
    dedalus_interpretation: Optional[DedalusInterpretation] = None


class ChunkProcessRequest(BaseModel):
    """Request for processing a 10-minute audio chunk."""
    chunk_index: int = Field(..., ge=0, description="Sequential chunk number")
    session_id: str = Field(..., description="Unique session identifier")
    patient_id: Optional[int] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(M|F|Other)$", description="Patient sex")


class ChunkProcessResponse(BaseModel):
    """Response from chunk processing."""
    chunk_index: int = Field(..., description="Processed chunk index")
    session_id: str = Field(..., description="Session identifier")
    cough_count: int = Field(0, description="Number of cough events in chunk")
    wheeze_windows: int = Field(0, description="Number of windows with wheeze")
    windows_processed: int = Field(..., description="Total windows processed (should be 600 for 10 minutes)")
    probability_timeline: ProbabilityTimeline = Field(default_factory=ProbabilityTimeline, description="Tile-level probabilities for the chunk")
    event_summary: EventSummary = Field(default_factory=EventSummary, description="Cough probability events in this chunk")
    detected_events: List[CoughEvent] = Field(default_factory=list, description="[Legacy] raw cough events for backward compatibility")


class FinalSummaryRequest(BaseModel):
    """Request for final nightly summary."""
    session_id: str = Field(..., description="Session identifier")
    symptom_form: Optional[SymptomForm] = Field(None, description="User symptom inputs for pattern panel")


class AudioAnalysisResponse(BaseModel):
    """Response model for audio analysis endpoint (development/testing only)."""
    patient_id: Optional[int] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    cough_count: int = Field(0, description="Total number of coughs detected")
    wheeze_count: int = Field(0, description="Total number of wheeze events detected")
    cough_probability_avg: float = Field(0.0, ge=0.0, le=1.0, description="Average cough probability")
    wheeze_probability_avg: float = Field(0.0, ge=0.0, le=1.0, description="Average wheeze probability")
    sleep_duration_minutes: Optional[float] = None
    detected_events: List[DetectedEvent] = Field(default_factory=list, description="All detected events with timestamps")
    dedalus_interpretation: Optional[DedalusInterpretation] = None
    processing_time_seconds: float = Field(..., description="Total processing time")
    windows_analyzed: int = Field(..., description="Number of 1-second windows analyzed")


# Rebuild all models to resolve forward references for OpenAPI schema generation
# This ensures all nested model references are properly resolved for FastAPI's OpenAPI schema
# Rebuild in dependency order (dependencies first, then dependents)

# Base models (no dependencies)
DetectedEvent.model_rebuild()
DedalusInterpretation.model_rebuild()
WindowPrediction.model_rebuild()
CoughEvent.model_rebuild()
SymptomForm.model_rebuild()
PatternScore.model_rebuild()
AttributeVector.model_rebuild()
AttributeVectorSeries.model_rebuild()
AttributeFlags.model_rebuild()
AttributePrevalence.model_rebuild()
ProbabilityEvent.model_rebuild()
EventSummary.model_rebuild()
ProbabilityTimeline.model_rebuild()
TrendComparison.model_rebuild()
QualityMetrics.model_rebuild()
DisplayStrings.model_rebuild()

# Models with dependencies
HourlyMetrics.model_rebuild()
NightlySummary.model_rebuild()
ChunkProcessResponse.model_rebuild()
AudioAnalysisResponse.model_rebuild()
MobileSummaryRequest.model_rebuild()
NightlySummaryResponse.model_rebuild()
ChunkProcessRequest.model_rebuild()
FinalSummaryRequest.model_rebuild()
AudioAnalysisRequest.model_rebuild()
