"""
FastAPI main application entrypoint.
Mobile-optimized pipeline: 20-30ms frames → 1s log-Mel windows → Binary models (Cough VAD + Wheeze).
"""
import logging
import time
import tempfile
import os
import base64
from pathlib import Path
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

from app.schemas import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    DetectedEvent,
    DedalusInterpretation,
    MobileSummaryRequest,
    NightlySummaryResponse,
    WindowPrediction,
    CoughEvent,
    CoughEventTag,
    ChunkProcessRequest,
    ChunkProcessResponse,
    FinalSummaryRequest,
    NightlySummary,
    SymptomForm,
    AttributePrevalence,
    HourlyMetrics,
    TrendComparison,
    QualityMetrics,
    DisplayStrings,
    ProbabilityTimeline,
    EventSummary,
    ProbabilityEvent,
)
from app.preprocessing.audio_clean import load_audio, trim_silence, denoise_audio, normalize_audio
from app.preprocessing.feature_extraction_mobile import prepare_mobile_features, segment_audio_1s_windows
from app.preprocessing.event_detection import (
    calculate_snr,
    build_probability_timeline,
    summarize_probability_events,
)
from app.models.cough_vad import CoughVAD
from app.models.wheeze_detector import WheezeDetector
import sys
from pathlib import Path

# Import CoughMultitaskCNN from training script
try:
    train_script_path = Path(__file__).parent.parent.parent / "train_cough_multitask.py"
    if train_script_path.exists():
        sys.path.insert(0, str(train_script_path.parent))
        from train_cough_multitask import CoughMultitaskCNN
        COUGH_MULTITASK_CNN_AVAILABLE = True
    else:
        COUGH_MULTITASK_CNN_AVAILABLE = False
        CoughMultitaskCNN = None
except Exception as e:
    # Logger not available yet, use print
    print(f"Warning: Could not import CoughMultitaskCNN: {e}")
    COUGH_MULTITASK_CNN_AVAILABLE = False
    CoughMultitaskCNN = None
from app.services.dedalus_client import DedalusClient
from app.services.pattern_panel import calculate_pattern_scores
from app.utils import setup_logging, ensure_directory, get_data_path
from datetime import datetime
from typing import Dict

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Respiratory Monitoring API",
    description="API for analyzing sleep respiratory audio with mobile-optimized ML models (1s windows, binary VAD)",
    version="2.0.0"
)

# CORS middleware for React Native frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models (lazy loading)
cough_vad: Optional[CoughVAD] = None
wheeze_detector: Optional[WheezeDetector] = None
dedalus_client: Optional[DedalusClient] = None

# Ensure data directory exists
ensure_directory(str(get_data_path()))

# In-memory session storage
# Structure: {session_id: SessionData}
sessions: Dict[str, Dict] = {}


class SessionData:
    """Session data structure for in-memory storage."""
    def __init__(self, session_id: str, patient_id: Optional[int] = None, 
                 age: Optional[int] = None, sex: Optional[str] = None):
        self.session_id = session_id
        self.patient_id = patient_id
        self.age = age
        self.sex = sex
        self.chunks: List[Dict] = []
        self.start_time = datetime.now()
        self.last_update = datetime.now()


def get_cough_vad() -> CoughVAD:
    """Get or initialize cough VAD."""
    global cough_vad
    if cough_vad is None:
        # Use CoughMultitaskCNN if available, otherwise use default
        if COUGH_MULTITASK_CNN_AVAILABLE:
            logger.info("✅ Using CoughMultitaskCNN model class (from train_cough_multitask.py)")
            # Create a factory function that returns the model instance
            def model_factory():
                return CoughMultitaskCNN(num_attrs=5)
            cough_vad = CoughVAD(model_class=model_factory)
        else:
            logger.warning("⚠️ CoughMultitaskCNN not available, using default SimpleCoughModel")
            cough_vad = CoughVAD()
    return cough_vad


def get_wheeze_detector() -> WheezeDetector:
    """Get or initialize wheeze detector."""
    global wheeze_detector
    if wheeze_detector is None:
        wheeze_detector = WheezeDetector()
    return wheeze_detector


def get_dedalus_client() -> DedalusClient:
    """Get or initialize Dedalus client."""
    global dedalus_client
    if dedalus_client is None:
        # Get OpenAI key from environment if available
        openai_key = os.getenv("OPENAI_API_KEY")
        dedalus_client = DedalusClient(openai_api_key=openai_key)
    return dedalus_client


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Sleep Respiratory Monitoring API",
        "version": "2.0.0",
        "pipeline": "mobile-optimized (1s log-Mel windows, binary VAD)"
    }


@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """
    Get status of a specific session (for debugging).
    
    Returns:
        Session information including chunk count and metadata
    """
    if session_id not in sessions:
        return {
            "session_id": session_id,
            "exists": False,
            "message": "Session not found. No chunks have been processed for this session yet."
        }
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "exists": True,
        "chunk_count": len(session["chunks"]),
        "patient_id": session.get("patient_id"),
        "age": session.get("age"),
        "sex": session.get("sex"),
        "start_time": session.get("start_time").isoformat() if session.get("start_time") else None,
        "last_update": session.get("last_update").isoformat() if session.get("last_update") else None,
        "chunks": [
            {
                "chunk_index": chunk.get("chunk_index"),
                "cough_count": chunk.get("cough_count", 0),
                "wheeze_windows": chunk.get("wheeze_windows", 0),
                "windows_processed": chunk.get("windows_processed", 0),
            }
            for chunk in session["chunks"]
        ]
    }


@app.get("/status")
async def get_status():
    """
    Get system status including model availability.
    
    Returns:
        Status information including which models are loaded and endpoint availability
    """
    cough_model = get_cough_vad()
    wheeze_model = get_wheeze_detector()
    dedalus_client = get_dedalus_client()
    
    return {
        "status": "healthy",
        "service": "Sleep Respiratory Monitoring API",
        "version": "2.0.0",
        "models": {
            "cough_model": {
                "loaded": cough_model.model is not None,
                "path": cough_model.model_path if cough_model.model_path else None,
                "num_outputs": cough_model.num_outputs,
                "input_shape": list(cough_model.input_shape) if cough_model.input_shape else None
            },
            "wheeze_detector": {
                "loaded": wheeze_model.model is not None,
                "path": wheeze_model.model_path if wheeze_model.model_path else None,
                "input_shape": list(wheeze_model.input_shape) if wheeze_model.input_shape else None
            }
        },
        "dedalus_ai": {
            "configured": dedalus_client.api_key is not None and dedalus_client.runner is not None,
            "model": dedalus_client.model if dedalus_client.runner else None
        },
        "endpoints": {
            "health": "GET /",
            "status": "GET /status",
            "session_status": "GET /session/{session_id}",
            "process_chunk": "POST /process-chunk",
            "final_summary": "POST /final-summary/{session_id}",
            "nightly_summary": "POST /nightly-summary",
            "analyze": "POST /analyze (dev/testing only)",
            "docs": "GET /docs"
        },
        "active_sessions": len(sessions)
    }


@app.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    patient_id: Optional[int] = Form(None),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    sleep_duration_minutes: Optional[float] = Form(None)
):
    """
    Analyze audio for respiratory anomalies and cough patterns.
    
    **Note: This endpoint is for development/testing only.**
    In production, the mobile app processes audio locally and sends summaries to /nightly-summary.
    
    Uses mobile-optimized pipeline:
    - 1-second log-Mel windows (from 20-30ms frames)
    - Binary models: Cough VAD and Wheeze Detector
    - Event-based detection with timestamps
    
    Args:
        audio_file: Audio file (WAV, MP3, M4A, etc.)
        patient_id: Optional patient identifier
        age: Optional patient age
        sex: Optional patient sex (M/F/Other)
        sleep_duration_minutes: Optional sleep duration in minutes
    
    Returns:
        AudioAnalysisResponse with detected events and interpretations
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        logger.info(f"Received audio file: {audio_file.filename}")
        
        # Save uploaded file temporarily
        temp_dir = get_data_path()
        temp_file_path = None
        
        try:
            # Create temporary file
            suffix = Path(audio_file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=temp_dir,
                suffix=suffix
            ) as temp_file:
                temp_file_path = temp_file.name
                # Write uploaded content
                content = await audio_file.read()
                temp_file.write(content)
                logger.info(f"Saved temporary file: {temp_file_path} ({len(content)} bytes)")
            
            # Preprocess audio for mobile pipeline (1-second windows)
            logger.info("Starting audio preprocessing for mobile pipeline...")
            audio, sample_rate = load_audio(temp_file_path, target_sr=16000)
            
            # Basic preprocessing (trim, denoise, normalize)
            # Use less aggressive trimming (top_db=30 instead of 40) to preserve quiet cough sounds
            audio = trim_silence(audio, sample_rate, top_db=30)
            audio = denoise_audio(audio, sample_rate)
            audio = normalize_audio(audio)
            
            # Segment into 1-second windows (mobile format)
            windows = segment_audio_1s_windows(audio, sr=sample_rate, window_length=1.0, stride=0.25)
            
            if not windows:
                raise HTTPException(status_code=400, detail="No audio windows extracted")
            
            # Initialize models
            cough_vad_model = get_cough_vad()
            wheeze_model = get_wheeze_detector()
            
            # Analyze each 1-second window
            logger.info(f"Analyzing {len(windows)} 1-second windows...")
            detected_events = []
            cough_count = 0
            wheeze_count = 0
            
            # Aggregate probabilities
            cough_probs = []
            wheeze_probs = []
            
            # Detection thresholds
            cough_threshold = 0.5  # Binary threshold for cough VAD
            wheeze_threshold = 0.5  # Binary threshold for wheeze
            
            for window_idx, (window_audio, start_time) in enumerate(windows):
                # Extract 1s log-Mel features (mobile format)
                features = prepare_mobile_features(
                    window_audio,
                    sr=sample_rate,
                    window_length=1.0
                )
                
                # Cough model (returns p_cough + attribute probability vector)
                p_cough, _attr_probs = cough_vad_model.predict(features)
                cough_prob = p_cough  # Use p_cough for threshold check
                cough_probs.append(cough_prob)
                
                if cough_prob >= cough_threshold:
                    cough_count += 1
                    detected_events.append(DetectedEvent(
                        event_type="cough",
                        timestamp=start_time,
                        probability=cough_prob,
                        window_index=window_idx
                    ))
                
                # Wheeze detection (binary)
                wheeze_prob = wheeze_model.predict(features)
                wheeze_probs.append(wheeze_prob)
                
                if wheeze_prob >= wheeze_threshold:
                    wheeze_count += 1
                    detected_events.append(DetectedEvent(
                        event_type="wheeze",
                        timestamp=start_time,
                        probability=wheeze_prob,
                        window_index=window_idx
                    ))
            
            # Aggregate probabilities (average)
            total_windows = len(windows)
            avg_cough_prob = sum(cough_probs) / total_windows if cough_probs else 0.0
            avg_wheeze_prob = sum(wheeze_probs) / total_windows if wheeze_probs else 0.0
            
            # Get Dedalus interpretation
            logger.info("Requesting Dedalus AI interpretation...")
            dedalus_client = get_dedalus_client()
            dedalus_interpretation = dedalus_client.interpret_results(
                cough_count=cough_count,
                cough_healthy_count=0,  # Not used in binary VAD
                cough_sick_count=0,  # Not used in binary VAD
                wheeze_probability=avg_wheeze_prob,
                crackle_probability=0.0,  # Not detected in binary model
                normal_probability=1.0 - max(avg_cough_prob, avg_wheeze_prob),
                sleep_duration_minutes=sleep_duration_minutes,
                patient_age=age,
                patient_sex=sex,
                wheeze_count=wheeze_count
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build response
            response = AudioAnalysisResponse(
                patient_id=patient_id,
                age=age,
                sex=sex,
                cough_count=cough_count,
                wheeze_count=wheeze_count,
                cough_probability_avg=round(avg_cough_prob, 3),
                wheeze_probability_avg=round(avg_wheeze_prob, 3),
                sleep_duration_minutes=sleep_duration_minutes,
                detected_events=detected_events,
                dedalus_interpretation=dedalus_interpretation,
                processing_time_seconds=round(processing_time, 2),
                windows_analyzed=len(windows)
            )
            
            logger.info(f"Analysis complete: {cough_count} coughs, {wheeze_count} wheezes, "
                       f"cough_prob={avg_cough_prob:.3f}, wheeze_prob={avg_wheeze_prob:.3f}, "
                       f"time={processing_time:.2f}s")
            
            return response
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Removed temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@app.post("/process-chunk", response_model=ChunkProcessResponse)
async def process_chunk(
    audio_chunk: UploadFile = File(..., description="10-minute audio chunk"),
    chunk_index: int = Form(..., description="Sequential chunk number"),
    session_id: str = Form(..., description="Unique session identifier"),
    patient_id: Optional[int] = Form(None),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None)
):
    """
    Process a 10-minute audio chunk.
    
    Receives 10-minute audio chunks throughout the night, processes them,
    and stores results in session storage for final aggregation.
    
    NOTE: This endpoint does NOT call Dedalus AI. Dedalus is only called
    once at the end of the night via the /final-summary/{session_id} endpoint.
    
    Args:
        audio_chunk: 10-minute audio file (600 seconds = 600 windows)
        chunk_index: Sequential chunk number (0, 1, 2, ...)
        session_id: Unique session identifier for the night
        patient_id: Optional patient identifier
        age: Optional patient age
        sex: Optional patient sex
    
    Returns:
        Chunk processing results with detected events
    """
    temp_file_path = None
    
    try:
        logger.info(f"📥 Received chunk {chunk_index} for session {session_id}")
        logger.info(f"   File: {audio_chunk.filename}, Content-Type: {audio_chunk.content_type}, Size: {audio_chunk.size if hasattr(audio_chunk, 'size') else 'unknown'}")
        
        # Validate session_id and chunk_index
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required and cannot be empty")
        
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail=f"chunk_index must be >= 0, got {chunk_index}")
        
        # Initialize or get session
        if session_id not in sessions:
            logger.info(f"Creating new session: {session_id}")
            sessions[session_id] = {
                "session_id": session_id,
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "chunks": [],
                "start_time": datetime.now(),
                "last_update": datetime.now()
            }
            logger.info(f"Session {session_id} created. Total sessions: {len(sessions)}")
        else:
            logger.debug(f"Session {session_id} already exists with {len(sessions[session_id]['chunks'])} chunks")
            # Update metadata if provided
            if patient_id is not None:
                sessions[session_id]["patient_id"] = patient_id
            if age is not None:
                sessions[session_id]["age"] = age
            if sex is not None:
                sessions[session_id]["sex"] = sex
            sessions[session_id]["last_update"] = datetime.now()
        
        # Validate and read uploaded file
        try:
            content = await audio_chunk.read()
            if not content or len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
            
            logger.info(f"   Read {len(content)} bytes from uploaded file")
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
        
        # Save uploaded file temporarily
        temp_dir = get_data_path()
        
        try:
            # Create temporary file with appropriate suffix
            suffix = Path(audio_chunk.filename).suffix if audio_chunk.filename else ".wav"
            # Ensure suffix is valid
            if not suffix or suffix not in [".wav", ".webm", ".mp3", ".m4a", ".ogg"]:
                suffix = ".wav"  # Default to WAV
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=temp_dir,
                suffix=suffix
            ) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(content)
                logger.info(f"   Saved temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create temporary file: {str(e)}")
        
        # Load and preprocess audio
        try:
            logger.info("   Loading audio file...")
            audio, sample_rate = load_audio(temp_file_path, target_sr=16000)
            logger.info(f"   Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to load audio file: {str(e)}. Ensure the file is a valid audio format (WAV, WebM, etc.)")
        
        try:
            logger.info("   Preprocessing audio (trim, denoise, normalize)...")
            # Use less aggressive trimming (top_db=30 instead of 40) to preserve quiet cough sounds
            audio = trim_silence(audio, sample_rate, top_db=30)
            audio = denoise_audio(audio, sample_rate)
            audio = normalize_audio(audio)
            logger.info(f"   Preprocessed audio: {len(audio)/sample_rate:.2f}s")
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to preprocess audio: {str(e)}")
        
        # Segment into 1-second windows
        try:
            logger.info("   Segmenting audio into 1s windows...")
            windows = segment_audio_1s_windows(audio, sr=sample_rate, window_length=1.0, stride=0.25)
            logger.info(f"   Created {len(windows)} windows")
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to segment audio: {str(e)}")
        
        if not windows:
            raise HTTPException(status_code=400, detail="No audio windows extracted from chunk. Audio may be too short or invalid.")
        
        # Initialize models
        try:
            logger.info("   Loading ML models...")
            cough_model = get_cough_vad()
            wheeze_model = get_wheeze_detector()
            
            # Check if models are actually loaded (not using mocks)
            if cough_model.model is None:
                logger.error("   ⚠️⚠️⚠️ COUGH MODEL NOT LOADED - USING MOCK PREDICTIONS! ⚠️⚠️⚠️")
                logger.error("   This means the model file is missing. Check model path in cough_vad.py")
            else:
                logger.info("   ✅ Cough model loaded successfully (using REAL model, not mocks)")
            
            if wheeze_model.model is None:
                logger.warning("   ⚠️ Wheeze model not loaded - using mock predictions")
            else:
                logger.info("   ✅ Wheeze model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load ML models: {str(e)}")
        
        # Calculate absolute window index offset from previous chunks
        # With stride=0.25s, we can't assume a fixed number of windows per chunk
        # Sum up windows from all previous chunks in this session
        total_previous_windows = 0
        if session_id in sessions:
            for prev_chunk in sessions[session_id]["chunks"]:
                if prev_chunk.get("chunk_index", -1) < chunk_index:
                    total_previous_windows += prev_chunk.get("windows_processed", 0)
        
        # Process each window
        try:
            logger.info(f"   Processing {len(windows)} windows...")
            logger.info(f"   Absolute window index offset: {total_previous_windows} (from {len(sessions[session_id]['chunks'])} previous chunks)")
            window_predictions = []
            wheeze_windows = 0
            
            for window_idx, (window_audio, start_time) in enumerate(windows):
                try:
                    # Calculate absolute window index across all chunks
                    # Each window represents 0.25s of audio (stride), so window index = total_previous + current
                    absolute_window_idx = total_previous_windows + window_idx
                    
                    # Calculate SNR for quality assessment
                    snr = calculate_snr(window_audio, sample_rate)
                    
                    # Extract features
                    features = prepare_mobile_features(
                        window_audio,
                        sr=sample_rate,
                        window_length=1.0
                    )
                    
                    # Run cough model (returns scalar + attribute vector)
                    p_cough, attr_probs = cough_model.predict(features)
                    
                    # Run wheeze model
                    p_wheeze = wheeze_model.predict(features)
                    if p_wheeze >= 0.5:
                        wheeze_windows += 1
                    
                    # Store window prediction
                    window_predictions.append(WindowPrediction(
                        window_index=absolute_window_idx,
                        p_cough=p_cough,
                        p_attr_wet=attr_probs.get("wet", 0.0),
                        p_attr_wheezing=attr_probs.get("wheezing", 0.0),
                        p_attr_stridor=attr_probs.get("stridor", 0.0),
                        p_attr_choking=attr_probs.get("choking", 0.0),
                        p_attr_congestion=attr_probs.get("congestion", 0.0),
                        p_wheeze=p_wheeze,
                        snr=snr
                    ))
                except Exception as e:
                    logger.warning(f"Error processing window {window_idx}: {e}", exc_info=True)
                    # Continue with other windows
                    continue
        except Exception as e:
            logger.error(f"Error in window processing loop: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process audio windows: {str(e)}")
        
        # Build probability timeline + event summary
        try:
            probability_timeline = build_probability_timeline(window_predictions)
            event_summary = summarize_probability_events(window_predictions)
            legacy_events = [
                _probability_event_to_cough_event(evt) for evt in event_summary.events
            ]
            logger.info(f"   Detected {event_summary.num_events} cough probability events")
        except Exception as e:
            logger.error(f"Error building probability event summary: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to build probability event summary: {str(e)}")
        
        # Store chunk results
        try:
            chunk_result = {
                "chunk_index": chunk_index,
                "cough_count": event_summary.num_events,
                "wheeze_windows": wheeze_windows,
                "windows_processed": len(windows),
                "probability_timeline": probability_timeline.model_dump(),
                "event_summary": event_summary.model_dump(),
                "detected_events": [e.model_dump() for e in legacy_events],
                "window_predictions": [w.model_dump() for w in window_predictions],
                "timestamp": datetime.now()
            }
            
            sessions[session_id]["chunks"].append(chunk_result)
            
            logger.info(f"✅ Chunk {chunk_index} processed: {event_summary.num_events} COUGH EVENTS DETECTED, {wheeze_windows} wheeze windows")
            logger.info(f"   📊 COUGH COUNT = {event_summary.num_events} (this is the NUMBER of coughs, not a percentage!)")
            logger.info(f"   Session {session_id} now has {len(sessions[session_id]['chunks'])} total chunks")
            if legacy_events:
                logger.info(f"   Event details: {[(e.start_ms/1000, e.end_ms/1000, e.confidence) for e in legacy_events[:5]]}")
        except Exception as e:
            logger.error(f"Error storing chunk results: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store chunk results: {str(e)}")
        
        # Return response
        try:
            return ChunkProcessResponse(
                chunk_index=chunk_index,
                session_id=session_id,
                cough_count=event_summary.num_events,
                wheeze_windows=wheeze_windows,
                windows_processed=len(windows),
                probability_timeline=probability_timeline,
                event_summary=event_summary,
                detected_events=legacy_events
            )
        except Exception as e:
            logger.error(f"Error creating response: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create response: {str(e)}")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they already have proper status codes)
        raise
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"❌ Unexpected error processing chunk {chunk_index} for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"   Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_file_path}: {e}")


@app.post("/final-summary/{session_id}", response_model=NightlySummary)
async def get_final_summary(
    session_id: str,
    symptom_form: Optional[SymptomForm] = Body(None, description="Optional symptom form for pattern panel")
):
    """
    Get final nightly summary aggregated from all chunks.
    
    Args:
        session_id: Session identifier
        symptom_form: Optional symptom form for pattern panel calculation
    
    Returns:
        Complete nightly summary with all metrics and pattern scores
    """
    try:
        # Auto-create session if it doesn't exist (in case chunks failed to register)
        if session_id not in sessions:
            logger.warning(f"Session {session_id} not found in storage. This may indicate that no chunks were successfully processed. Creating empty session.")
            sessions[session_id] = {
                "session_id": session_id,
                "patient_id": None,
                "age": None,
                "sex": None,
                "chunks": [],
                "start_time": datetime.now(),
                "last_update": datetime.now()
            }
        
        session = sessions[session_id]
        chunks = session["chunks"]
        
        if not chunks:
            logger.warning(f"Session {session_id} exists but has no processed chunks. This may indicate that chunks failed to process or were not received.")
            raise HTTPException(
                status_code=400, 
                detail=f"No chunks processed for session {session_id}. Please ensure chunks were successfully sent to /process-chunk endpoint."
            )
        
        # Aggregate all events and window predictions
        all_cough_events = []
        all_window_predictions = []
        total_wheeze_windows = 0
        total_windows = 0
        
        for chunk in chunks:
            # Reconstruct events from stored data
            for event_dict in chunk["detected_events"]:
                all_cough_events.append(CoughEvent(**event_dict))
            
            # Reconstruct window predictions
            for window_dict in chunk.get("window_predictions", []):
                all_window_predictions.append(WindowPrediction(**window_dict))
            
            total_wheeze_windows += chunk["wheeze_windows"]
            total_windows += chunk["windows_processed"]
        
        # Build probability timeline + nightly event summary
        probability_timeline = build_probability_timeline(all_window_predictions)
        nightly_event_summary = summarize_probability_events(all_window_predictions)
        all_cough_events = [
            _probability_event_to_cough_event(evt) for evt in nightly_event_summary.events
        ]
        
        # Calculate metrics
        total_duration_minutes = total_windows / 60.0
        coughs_per_hour = (len(all_cough_events) / total_duration_minutes) * 60.0 if total_duration_minutes > 0 else 0.0
        wheeze_time_percent = (total_wheeze_windows / total_windows) * 100.0 if total_windows > 0 else 0.0
        
        # Calculate bout count and lengths
        bout_count, bout_lengths, inter_cough_intervals = _calculate_bouts(all_cough_events)
        
        # Calculate additional cough metrics
        avg_bout_length = sum(bout_lengths) / len(bout_lengths) if bout_lengths else None
        avg_inter_cough_interval = sum(inter_cough_intervals) / len(inter_cough_intervals) if inter_cough_intervals else None
        
        # Calculate max coughs in single hour
        hourly_breakdown = _calculate_hourly_breakdown(all_cough_events, all_window_predictions, total_duration_minutes)
        max_coughs_in_single_hour = max((h.cough_count for h in hourly_breakdown), default=0)
        
        # Calculate wheeze metrics
        wheeze_periods = []
        current_wheeze_start = None
        for i, window in enumerate(all_window_predictions):
            if window.p_wheeze >= 0.5:
                if current_wheeze_start is None:
                    current_wheeze_start = i
            else:
                if current_wheeze_start is not None:
                    # Duration in seconds (each window is 1 second)
                    wheeze_periods.append(i - current_wheeze_start)
                    current_wheeze_start = None
        if current_wheeze_start is not None:
            # Duration in seconds
            wheeze_periods.append(len(all_window_predictions) - current_wheeze_start)
        
        longest_wheeze_duration = max(wheeze_periods, default=0) if wheeze_periods else None
        wheeze_intensity_avg = None
        if wheeze_periods:
            wheeze_windows = [w for w in all_window_predictions if w.p_wheeze >= 0.5]
            if wheeze_windows:
                wheeze_intensity_avg = sum(w.p_wheeze for w in wheeze_windows) / len(wheeze_windows)
        
        # Calculate attribute prevalence
        attribute_prevalence = _calculate_attribute_prevalence(all_window_predictions, all_cough_events)
        
        # Calculate pattern scores
        pattern_scores = []
        if symptom_form:
            pattern_scores = calculate_pattern_scores(
                coughs_per_hour,
                wheeze_time_percent,
                attribute_prevalence,
                symptom_form
            )
        
        # Get Dedalus interpretation (ONLY called once at end of night, not per chunk)
        dedalus_interpretation = None
        try:
            logger.info(f"Calling Dedalus AI for final interpretation (session {session_id})")
            dedalus_client = get_dedalus_client()
            dedalus_interpretation = dedalus_client.interpret_results(
                cough_count=len(all_cough_events),
                wheeze_count=total_wheeze_windows,
                wheeze_probability=wheeze_time_percent / 100.0,
                cough_healthy_count=0,  # Not used in binary VAD
                cough_sick_count=0,  # Not used in binary VAD
                crackle_probability=0.0,  # Not detected in binary model
                normal_probability=max(
                    0.0, 1.0 - (wheeze_time_percent / 100.0)
                ),
                sleep_duration_minutes=total_duration_minutes,
                patient_age=session.get("age"),
                patient_sex=session.get("sex"),
                attribute_wet_percent=attribute_prevalence.wet,
                attribute_stridor_percent=attribute_prevalence.stridor,
                attribute_choking_percent=attribute_prevalence.choking,
                attribute_congestion_percent=attribute_prevalence.congestion,
                attribute_wheezing_selfreport_percent=attribute_prevalence.wheezing
            )
        except Exception as dedalus_error:
            logger.warning(f"Dedalus interpretation failed for session {session_id}: {dedalus_error}")
        
        # Calculate quality metrics
        quality_metrics = _calculate_quality_metrics(all_window_predictions, all_cough_events)
        
        # Calculate display strings
        display_strings = _calculate_display_strings(
            total_duration_minutes,
            coughs_per_hour,
            dedalus_interpretation.severity if dedalus_interpretation else None,
            quality_metrics.quality_score
        )
        
        # Calculate trend comparison (placeholder - requires historical data storage)
        current_metrics = {
            "coughs_per_hour": coughs_per_hour,
            "wheeze_time_percent": wheeze_time_percent,
            "bout_count": float(bout_count)
        }
        trend_comparison = _calculate_trend_comparison(current_metrics, None)  # TODO: Pass historical data
        
        # Calculate trend arrow (simplified - would need historical data for real calculation)
        trend_arrow = None  # TODO: Implement with historical data
        
        # Build summary
        summary = NightlySummary(
            session_id=session_id,
            patient_id=session.get("patient_id"),
            age=session.get("age"),
            sex=session.get("sex"),
            total_duration_minutes=total_duration_minutes,
            coughs_per_hour=coughs_per_hour,
            bout_count=bout_count,
            bout_lengths=bout_lengths,
            inter_cough_intervals=inter_cough_intervals,
            avg_bout_length_seconds=avg_bout_length,
            avg_inter_cough_interval_seconds=avg_inter_cough_interval,
            max_coughs_in_single_hour=max_coughs_in_single_hour,
            wheeze_time_percent=wheeze_time_percent,
            longest_wheeze_duration_seconds=longest_wheeze_duration,
            wheeze_intensity_avg=wheeze_intensity_avg,
            attribute_prevalence=attribute_prevalence,
            cough_events=all_cough_events,
            event_summary=nightly_event_summary,
            probability_timeline=probability_timeline,
            pattern_scores=pattern_scores,
            trend_arrow=trend_arrow,
            hourly_breakdown=hourly_breakdown,
            trend_comparison=trend_comparison,
            quality_metrics=quality_metrics,
            display_strings=display_strings,
            dedalus_interpretation=dedalus_interpretation
        )
        
        logger.info(f"Final summary generated for session {session_id}: {len(all_cough_events)} coughs, {wheeze_time_percent:.1f}% wheeze")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating final summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _calculate_bouts(cough_events: List[CoughEvent]) -> tuple[int, List[float], List[float]]:
    """Calculate bout count, lengths, and inter-cough intervals."""
    if not cough_events:
        return 0, [], []
    
    # Sort events by start time
    sorted_events = sorted(cough_events, key=lambda e: e.start_ms)
    
    # Group events into bouts (events within 30 seconds are in same bout)
    bouts = []
    current_bout = [sorted_events[0]]
    
    for event in sorted_events[1:]:
        time_since_last = (event.start_ms - current_bout[-1].end_ms) / 1000.0  # seconds
        if time_since_last <= 30.0:  # 30 second threshold
            current_bout.append(event)
        else:
            bouts.append(current_bout)
            current_bout = [event]
    
    if current_bout:
        bouts.append(current_bout)
    
    # Calculate bout lengths
    bout_lengths = []
    for bout in bouts:
        bout_start = bout[0].start_ms
        bout_end = bout[-1].end_ms
        bout_lengths.append((bout_end - bout_start) / 1000.0)  # seconds
    
    # Calculate inter-cough intervals
    inter_cough_intervals = []
    for i in range(len(sorted_events) - 1):
        interval = (sorted_events[i + 1].start_ms - sorted_events[i].end_ms) / 1000.0  # seconds
        if interval > 0:
            inter_cough_intervals.append(interval)
    
    return len(bouts), bout_lengths, inter_cough_intervals


def _calculate_attribute_prevalence(
    window_predictions: List[WindowPrediction],
    cough_events: List[CoughEvent]
) -> AttributePrevalence:
    """Calculate attribute prevalence from windows and events."""
    if not window_predictions:
        logger.warning("No window predictions available for attribute prevalence calculation")
        return AttributePrevalence()
    
    # Calculate from windows (tiles)
    total_windows = len(window_predictions)
    logger.info(f"Calculating attribute prevalence from {total_windows} window predictions")
    
    # Log sample attribute probabilities to debug
    if window_predictions:
        sample = window_predictions[0]
        logger.info(f"Sample window prediction: p_attr_wet={sample.p_attr_wet:.3f}, p_attr_stridor={sample.p_attr_stridor:.3f}, p_attr_choking={sample.p_attr_choking:.3f}, p_attr_congestion={sample.p_attr_congestion:.3f}, p_attr_wheezing={sample.p_attr_wheezing:.3f}")
    
    # Count windows with attribute probabilities >= 0.5
    wet_count = sum(1 for w in window_predictions if w.p_attr_wet >= 0.5)
    stridor_count = sum(1 for w in window_predictions if w.p_attr_stridor >= 0.5)
    choking_count = sum(1 for w in window_predictions if w.p_attr_choking >= 0.5)
    congestion_count = sum(1 for w in window_predictions if w.p_attr_congestion >= 0.5)
    wheezing_count = sum(1 for w in window_predictions if w.p_attr_wheezing >= 0.5)
    
    # Log sample values to debug
    if window_predictions:
        sample_attrs = window_predictions[0]
        logger.info(f"🔍 Sample window attribute values: wet={sample_attrs.p_attr_wet:.3f}, stridor={sample_attrs.p_attr_stridor:.3f}, choking={sample_attrs.p_attr_choking:.3f}, congestion={sample_attrs.p_attr_congestion:.3f}, wheezing={sample_attrs.p_attr_wheezing:.3f}")
        
        # Check if all values are 1.0 (which would indicate a problem)
        all_wet = [w.p_attr_wet for w in window_predictions[:10]]
        logger.info(f"🔍 First 10 p_attr_wet values: {all_wet}")
    
    # Also calculate average probabilities (not just threshold counts)
    avg_wet = sum(w.p_attr_wet for w in window_predictions) / total_windows if total_windows > 0 else 0.0
    avg_stridor = sum(w.p_attr_stridor for w in window_predictions) / total_windows if total_windows > 0 else 0.0
    avg_choking = sum(w.p_attr_choking for w in window_predictions) / total_windows if total_windows > 0 else 0.0
    avg_congestion = sum(w.p_attr_congestion for w in window_predictions) / total_windows if total_windows > 0 else 0.0
    avg_wheezing = sum(w.p_attr_wheezing for w in window_predictions) / total_windows if total_windows > 0 else 0.0
    
    logger.info(f"Attribute counts (>=0.5): wet={wet_count}, stridor={stridor_count}, choking={choking_count}, congestion={congestion_count}, wheezing={wheezing_count}")
    logger.info(f"Average attribute probabilities: wet={avg_wet:.3f}, stridor={avg_stridor:.3f}, choking={avg_choking:.3f}, congestion={avg_congestion:.3f}, wheezing={avg_wheezing:.3f}")
    
    # Check if averages are suspiciously high (all 1.0)
    if avg_wet >= 0.99 and avg_stridor >= 0.99 and avg_choking >= 0.99 and avg_congestion >= 0.99 and avg_wheezing >= 0.99:
        logger.error(f"⚠️⚠️⚠️ ALL ATTRIBUTE PROBABILITIES ARE ~1.0! This indicates a problem with model output processing. Check if model is outputting logits that need sigmoid, or if values are being clamped incorrectly.")
        logger.error(f"   Sample window values: {sample_attrs.p_attr_wet:.3f}, {sample_attrs.p_attr_stridor:.3f}, {sample_attrs.p_attr_choking:.3f}, {sample_attrs.p_attr_congestion:.3f}, {sample_attrs.p_attr_wheezing:.3f}")
    
    # Calculate percentages - use AVERAGE PROBABILITIES (not threshold counts)
    # This gives the actual percentage from model output, not just binary threshold
    result = AttributePrevalence(
        wet=avg_wet * 100.0,  # Convert probability (0-1) to percentage (0-100)
        wheezing=avg_wheezing * 100.0,
        stridor=avg_stridor * 100.0,
        choking=avg_choking * 100.0,
        congestion=avg_congestion * 100.0
    )
    
    logger.info(f"Attribute prevalence result: wet={result.wet:.1f}%, wheezing={result.wheezing:.1f}%, stridor={result.stridor:.1f}%, choking={result.choking:.1f}%, congestion={result.congestion:.1f}%")
    
    return result


def _probability_event_to_cough_event(prob_event: ProbabilityEvent) -> CoughEvent:
    """Convert probability-based event to legacy CoughEvent for downstream metrics."""
    tags: List[CoughEventTag] = []
    if prob_event.attr_flags.wet:
        tags.append(CoughEventTag.WET)
    if prob_event.attr_flags.stridor:
        tags.append(CoughEventTag.STRIDOR)
    if prob_event.attr_flags.choking:
        tags.append(CoughEventTag.CHOKING)
    if prob_event.attr_flags.congestion:
        tags.append(CoughEventTag.CONGESTION)
    if prob_event.attr_flags.wheezing:
        tags.append(CoughEventTag.SELFREPORTED_WHEEZING)
    if not tags:
        tags.append(CoughEventTag.COUGH_UNSPEC)
    
    return CoughEvent(
        start_ms=int(prob_event.start * 1000),
        end_ms=int(prob_event.end * 1000),
        confidence=float(prob_event.p_cough_max),
        tags=tags,
        quality_flag=None,
        window_indices=prob_event.tile_indices,
    )


def _calculate_hourly_breakdown(
    cough_events: List[CoughEvent],
    window_predictions: List[WindowPrediction],
    total_duration_minutes: float
) -> List[HourlyMetrics]:
    """Calculate hourly breakdown of metrics."""
    hourly_data: Dict[int, Dict] = {}
    
    # Initialize all hours that have data
    max_hour = int(total_duration_minutes / 60.0)
    for hour in range(max_hour + 1):
        hourly_data[hour] = {
            "cough_count": 0,
            "wheeze_windows": 0,
            "total_windows": 0,
            "events": []
        }
    
    # Process cough events by hour (start_ms is in milliseconds)
    for event in cough_events:
        hour = int((event.start_ms / 1000.0) / 3600.0)
        if hour in hourly_data:
            hourly_data[hour]["cough_count"] += 1
            hourly_data[hour]["events"].append(event)
    
    # Process window predictions by hour
    # window_index represents the 1-second window number (0, 1, 2, ...)
    for window in window_predictions:
        hour = int(window.window_index / 3600.0)  # 3600 windows = 1 hour
        if hour in hourly_data:
            hourly_data[hour]["total_windows"] += 1
            if window.p_wheeze >= 0.5:
                hourly_data[hour]["wheeze_windows"] += 1
    
    # Build hourly metrics
    hourly_metrics = []
    for hour in sorted(hourly_data.keys()):
        data = hourly_data[hour]
        wheeze_percent = (data["wheeze_windows"] / data["total_windows"] * 100.0) if data["total_windows"] > 0 else 0.0
        
        hourly_metrics.append(HourlyMetrics(
            hour=hour,
            cough_count=data["cough_count"],
            wheeze_percent=wheeze_percent,
            events=data["events"]
        ))
    
    return hourly_metrics


def _calculate_trend_comparison(
    current_metrics: Dict[str, float],
    historical_data: Optional[Dict] = None
) -> Optional[TrendComparison]:
    """
    Calculate trend comparisons vs historical data.
    
    Note: This is a placeholder. In production, historical data should be retrieved
    from a database. For now, returns None if no historical data is available.
    """
    if not historical_data:
        return None
    
    vs_last_night = {}
    vs_7_day_avg = {}
    percent_changes = {}
    
    # Compare vs last night
    if "last_night" in historical_data:
        last_night = historical_data["last_night"]
        for key in current_metrics:
            if key in last_night:
                change = current_metrics[key] - last_night[key]
                vs_last_night[key] = change
                if last_night[key] != 0:
                    percent_changes[f"{key}_vs_last_night"] = (change / last_night[key]) * 100.0
    
    # Compare vs 7-day average
    if "seven_day_avg" in historical_data:
        seven_day_avg = historical_data["seven_day_avg"]
        for key in current_metrics:
            if key in seven_day_avg:
                change = current_metrics[key] - seven_day_avg[key]
                vs_7_day_avg[key] = change
                if seven_day_avg[key] != 0:
                    percent_changes[f"{key}_vs_7day"] = (change / seven_day_avg[key]) * 100.0
    
    return TrendComparison(
        vs_last_night=vs_last_night,
        vs_7_day_avg=vs_7_day_avg,
        percent_changes=percent_changes
    )


def _calculate_quality_metrics(
    window_predictions: List[WindowPrediction],
    cough_events: List[CoughEvent]
) -> QualityMetrics:
    """Calculate audio quality and reliability metrics."""
    # Calculate average SNR
    snr_values = [w.snr for w in window_predictions if w.snr is not None]
    avg_snr = sum(snr_values) / len(snr_values) if snr_values else 0.0
    
    # Count low quality periods (SNR < 10 dB)
    low_quality_periods = sum(1 for w in window_predictions if w.snr is not None and w.snr < 10.0)
    
    # Count high confidence events
    high_confidence_events = sum(1 for e in cough_events if e.confidence > 0.8)
    
    # Count suppressed events
    suppressed_events = sum(1 for e in cough_events if e.quality_flag == "INSUFFICIENT_SIGNAL")
    
    # Calculate overall quality score (0-100)
    # Based on SNR (0-30 dB maps to 0-100), with penalties for low quality periods
    snr_score = min(100.0, max(0.0, (avg_snr / 30.0) * 100.0))
    quality_penalty = min(30.0, (low_quality_periods / max(1, len(window_predictions))) * 100.0)
    quality_score = max(0.0, snr_score - quality_penalty)
    
    return QualityMetrics(
        avg_snr=avg_snr,
        quality_score=quality_score,
        low_quality_periods_count=low_quality_periods,
        high_confidence_events_count=high_confidence_events,
        suppressed_events_count=suppressed_events
    )


def _calculate_display_strings(
    total_duration_minutes: float,
    coughs_per_hour: float,
    severity: Optional[str],
    quality_score: float
) -> DisplayStrings:
    """Generate formatted strings for UI display."""
    # Format sleep duration
    hours = int(total_duration_minutes // 60)
    minutes = int(total_duration_minutes % 60)
    if hours > 0 and minutes > 0:
        sleep_duration_formatted = f"{hours}h {minutes}m"
    elif hours > 0:
        sleep_duration_formatted = f"{hours}h"
    else:
        sleep_duration_formatted = f"{minutes}m"
    
    # Format coughs per hour
    coughs_per_hour_formatted = f"{coughs_per_hour:.1f} /hr"
    
    # Determine severity badge color
    if severity:
        severity_lower = severity.lower()
        if "severe" in severity_lower:
            severity_badge_color = "red"
        elif "moderate" in severity_lower:
            severity_badge_color = "yellow"
        else:
            severity_badge_color = "green"
    else:
        # Default based on metrics if no severity from Dedalus
        if coughs_per_hour > 10 or quality_score < 50:
            severity_badge_color = "red"
        elif coughs_per_hour > 5 or quality_score < 70:
            severity_badge_color = "yellow"
        else:
            severity_badge_color = "green"
    
    # Calculate overall quality score (health quality, not just audio quality)
    # Combines cough frequency, wheeze, and quality metrics
    # Lower is better (0 = perfect health, 100 = worst)
    health_quality = min(100.0, max(0.0, 
        (coughs_per_hour * 5.0) +  # Cough penalty
        (100.0 - quality_score) * 0.3  # Quality penalty
    ))
    overall_quality_score = 100.0 - health_quality  # Invert so higher is better
    
    return DisplayStrings(
        sleep_duration_formatted=sleep_duration_formatted,
        coughs_per_hour_formatted=coughs_per_hour_formatted,
        severity_badge_color=severity_badge_color,
        overall_quality_score=overall_quality_score
    )


@app.post("/nightly-summary", response_model=NightlySummaryResponse)
async def create_nightly_summary(
    summary: MobileSummaryRequest = Body(..., description="Summary from mobile app after local processing")
):
    """
    Create nightly summary from mobile app processing results.
    
    This endpoint accepts a pre-computed summary from the mobile app (which processes
    audio locally) and generates a Dedalus AI interpretation.
    
    The mobile app handles all audio processing locally:
    - Audio recording and buffering
    - Preprocessing (normalization, noise reduction)
    - Feature extraction (log-Mel spectrograms)
    - ONNX model inference (Cough VAD + Wheeze Detector)
    - Event detection and aggregation
    
    This backend endpoint only:
    - Validates the summary
    - Calls Dedalus AI for health interpretation
    - Returns interpretation results
    
    Args:
        summary: Pre-computed summary from mobile app with detected events and statistics
    
    Returns:
        Dedalus AI interpretation and optional summary ID
    """
    try:
        logger.info(f"Received nightly summary: patient_id={summary.patient_id}, "
                   f"cough_count={summary.cough_count}, wheeze_count={summary.wheeze_count}")
        
        # Get Dedalus client
        dedalus_client = get_dedalus_client()
        
        # Call Dedalus AI with summary data
        # Note: MobileSummaryRequest doesn't include attribute prevalence, so we pass 0.0
        # In production, mobile app should include attribute_prevalence in the summary
        dedalus_interpretation = dedalus_client.interpret_results(
            cough_count=summary.cough_count,
            wheeze_count=summary.wheeze_count,
            wheeze_probability=summary.wheeze_probability_avg,
            sleep_duration_minutes=summary.sleep_duration_minutes,
            patient_age=summary.age,
            patient_sex=summary.sex,
            cough_healthy_count=0,  # Not used in binary VAD model
            cough_sick_count=0,  # Not used in binary VAD model
            crackle_probability=0.0,  # Not detected in binary model
            normal_probability=1.0 - max(summary.cough_probability_avg, summary.wheeze_probability_avg),
            attribute_wet_percent=0.0,  # Not available in MobileSummaryRequest
            attribute_stridor_percent=0.0,
            attribute_choking_percent=0.0,
            attribute_congestion_percent=0.0,
            attribute_wheezing_selfreport_percent=0.0
        )
        
        # TODO: Optionally store summary in database
        # summary_id = store_summary(summary, dedalus_interpretation)
        summary_id = None
        
        return NightlySummaryResponse(
            dedalus_interpretation=dedalus_interpretation,
            summary_id=summary_id
        )
        
    except Exception as e:
        logger.error(f"Error creating nightly summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
