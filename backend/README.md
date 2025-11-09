# Sleep Respiratory Monitoring Backend

**Note: This backend processes 10-minute audio chunks throughout the night and provides health interpretation.**

The backend receives 10-minute audio chunks, processes them with ML models, detects cough events with attributes, and provides pattern-based health insights.

## Architecture

### Mobile App (Primary Processing)
- **Audio Recording**: Captures continuous audio stream
- **Buffering**: Breaks audio into 10 minutechunks
- **Preprocessing**: Normalization, noise reduction, resampling
- **Feature Extraction**: MFCCs, Mel spectrograms
- **PyTorch Inference**: Runs Cough VAD + Wheeze Detector models locally
- **Classification**: Maps model outputs to labels
- **Result Handling**: Displays results in UI, stores locally

### Backend (Processing Service)
- **Chunk Processing**: Receives and processes 10-minute audio chunks
- **Event Detection**: Detects cough events with hysteresis merging and attribute tagging
- **Pattern Panel**: Fuses audio metrics with symptom forms to generate pattern scores
- **Dedalus AI Integration**: Optional health interpretation
- **Development/Testing**: `/analyze` endpoint for testing with complete audio files

## Features

- **10-Minute Chunk Processing**: Receives audio chunks throughout the night
- **Attribute-Based Detection**: Cough model outputs 6 probabilities (cough + 5 attributes)
- **Event Detection with Hysteresis**: Merges consecutive detections into events
- **Pattern Panel**: Fuses audio metrics with symptoms to generate pattern scores (asthma-like, COPD-like, etc.)
- **Dedalus AI Integration**: Optional health interpretation
- **Development Tools**: `/analyze` endpoint for testing with complete audio files

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entrypoint
│   ├── schemas.py            # Pydantic request/response models
│   ├── utils.py              # Helper functions
│   ├── models/               # ML model loading scripts
│   │   ├── __init__.py
│   │   ├── cough_vad.py      # Cough model with attributes
│   │   └── wheeze_detector.py # Wheeze detector
│   ├── preprocessing/        # Audio cleaning + feature extraction
│   │   ├── __init__.py
│   │   ├── audio_clean.py
│   │   ├── feature_extraction_mobile.py
│   │   └── event_detection.py
│   └── services/
│       ├── __init__.py
│       ├── dedalus_client.py
│       └── pattern_panel.py
├── assets/                   # Pretrained PyTorch models (place models here)
├── data/                     # Temporary audio storage
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the backend directory:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   Create a `.env` file in the backend directory:
   ```env
   DEDALUS_API_KEY=your_api_key_here
   ```

5. **Add ML models** (required for production, optional for development):
   Place your pre-trained PyTorch models in the `assets/` directory:
   - `assets/cough_model.pt` - Cough model with 6 outputs (p_cough + 5 attributes)
   - `assets/wheeze_detector.pt` - Binary wheeze detector
   
   **Note**: The system will work with mock predictions if models are not provided, which is useful for development and testing.
   
   **Model format**: Models can be saved as:
   - Full model: `torch.save(model, 'model.pt')`
   - State dict: `torch.save({'model_state_dict': model.state_dict()}, 'model.pt')`
   - If using state dict, you'll need to provide the model class when initializing
   
   **Verify model loading**: After starting the server, check `GET /status` to see if models are loaded correctly.

6. **Verify setup**:
   ```bash
   # Start the server
   python run.py
   
   # In another terminal, run the test script
   python test_api.py
   ```
   
   This will verify:
   - Server is running
   - Models are loaded (or using mocks)
   - All endpoints are accessible
   - API documentation is available

## Running the Server

### Development Mode

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### GET `/`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "Sleep Respiratory Monitoring API",
  "version": "1.0.0"
}
```

### POST `/process-chunk`

**Process 10-minute audio chunks throughout the night.**

Receives 10-minute audio chunks, processes them into 1-second windows, runs ML models, detects events, and stores results in session.

**Request** (multipart/form-data):
- `audio_chunk` (file): 10-minute audio file
- `chunk_index` (int): Sequential chunk number (0, 1, 2, ...)
- `session_id` (str): Unique session identifier
- `patient_id` (int, optional): Patient identifier
- `age` (int, optional): Patient age
- `sex` (str, optional): Patient sex

**Response**:
```json
{
  "chunk_index": 0,
  "session_id": "uuid",
  "cough_count": 5,
  "wheeze_windows": 12,
  "windows_processed": 600,
  "detected_events": [...]
}
```

### POST `/final-summary/{session_id}`

**Get final nightly summary aggregated from all chunks.**

Aggregates all processed chunks, calculates metrics, generates pattern scores, and returns complete summary.

**Request** (JSON body, optional):
```json
{
  "symptom_form": {
    "fever": false,
    "sore_throat": true,
    "chest_tightness": false,
    "duration": 3,
    "nocturnal_worsening": true,
    "asthma_history": false,
    "copd_history": false,
    "age_band": "31-50",
    "smoker": false
  }
}
```

**Response**: Complete `NightlySummary` with all metrics, events, pattern scores, and Dedalus interpretation.

### POST `/nightly-summary`

**Legacy endpoint.** Accepts pre-computed summary from mobile app and returns Dedalus AI interpretation.

**Request** (JSON body):
```json
{
  "patient_id": 101,
  "age": 63,
  "sex": "M",
  "sleep_duration_minutes": 420,
  "cough_count": 15,
  "wheeze_count": 3,
  "cough_probability_avg": 0.45,
  "wheeze_probability_avg": 0.12,
  "detected_events": [
    {
      "event_type": "cough",
      "timestamp": 12.5,
      "probability": 0.82,
      "window_index": 12
    },
    {
      "event_type": "wheeze",
      "timestamp": 45.2,
      "probability": 0.65,
      "window_index": 45
    }
  ]
}
```

**Response**:
```json
{
  "dedalus_interpretation": {
    "interpretation": "Detected frequent coughing (15 cough events) — significant wheeze patterns detected.",
    "severity": "moderate",
    "recommendations": [
      "Consider consulting a healthcare provider if persistent",
      "Wheezing may indicate airway inflammation or obstruction"
    ]
  },
  "summary_id": null
}
```

### POST `/analyze`

**Development/Testing Only.** Analyze complete audio file for testing purposes.

**Note**: In production, the mobile app processes audio locally and sends summaries to `/nightly-summary`.

**Request**:
- `audio_file` (file, required): Audio file (WAV, MP3, M4A, etc.)
- `patient_id` (int, optional): Patient identifier
- `age` (int, optional): Patient age
- `sex` (string, optional): Patient sex (M/F/Other)
- `sleep_duration_minutes` (float, optional): Sleep duration in minutes

**Response**: Similar to `/nightly-summary` but includes processing time and windows analyzed.

## Testing the API

### Using curl

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "audio_file=@path/to/your/audio.wav" \
  -F "patient_id=101" \
  -F "age=63" \
  -F "sex=M" \
  -F "sleep_duration_minutes=420"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/analyze"
files = {"audio_file": open("path/to/audio.wav", "rb")}
data = {
    "patient_id": 101,
    "age": 63,
    "sex": "M",
    "sleep_duration_minutes": 420
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Using the Interactive Docs

Navigate to http://localhost:8000/docs and use the interactive Swagger UI to test the endpoints.

## Mobile App Processing Pipeline

The mobile app handles all audio processing locally:

1. **Recording**: Captures continuous audio stream (16kHz mono)
2. **Buffering**: Breaks into 2-3 second chunks
3. **Preprocessing**: Normalization, noise reduction, resampling
4. **Feature Extraction**: Log-Mel spectrograms (1-second windows)
5. **PyTorch Inference**: Runs Cough VAD + Wheeze Detector models locally
6. **Event Detection**: Detects coughs and wheezes with timestamps
7. **Aggregation**: Creates summary with counts and probabilities
8. **Backend Call**: Sends summary to `/nightly-summary` for Dedalus AI interpretation

## Backend Processing (Development Only)

The `/analyze` endpoint processes complete audio files for testing:

1. **Load**: Audio is loaded and resampled to 16kHz mono
2. **Trim**: Silence is removed from beginning and end
3. **Denoise**: Spectral noise reduction is applied
4. **Normalize**: Audio amplitude is normalized to -20dB
5. **Segment**: Audio is split into 1-second windows
6. **Feature Extraction**: Log-Mel spectrograms are extracted
7. **Inference**: Binary models (Cough VAD + Wheeze Detector) analyze each window
8. **Aggregation**: Results are aggregated across all windows
9. **Interpretation**: Structured data is sent to Dedalus AI
10. **Response**: JSON response with all detected events and interpretations

## ML Models

### Model Format

The system expects PyTorch models (`.pt` files). Models should:
- Accept input shape: `(batch, channels, height, width)` or `(batch, height, width)`
- Output class probabilities (logits or probabilities)
- Be saved as full models or state dicts

### Model Requirements

The system expects pre-trained PyTorch models:

- **Cough Model** (`assets/cough_model.pt`): Outputs 6 values per window
  - `p_cough`: Cough probability [0, 1]
  - `p_attr_wet`: Wet cough attribute [0, 1]
  - `p_attr_stridor`: Stridor attribute [0, 1]
  - `p_attr_choking`: Choking attribute [0, 1]
  - `p_attr_congestion`: Congestion attribute [0, 1]
  - `p_attr_wheezing_selfreport`: Self-reported wheezing attribute [0, 1]

- **Wheeze Model** (`assets/wheeze_detector.pt`): Binary output
  - Wheeze probability [0, 1]

### Model Loading

Models can be saved in different formats:
- **Full model**: `torch.save(model, 'model.pt')` - Loads directly
- **State dict**: `torch.save({'model_state_dict': model.state_dict()}, 'model.pt')` - Requires model class
- **Direct state dict**: `torch.save(model.state_dict(), 'model.pt')` - Requires model class

If loading a state dict, you'll need to provide the model architecture class when initializing the model loaders.

## Dedalus AI Integration

The system integrates with Dedalus AI for health interpretation. Set the `DEDALUS_API_KEY` environment variable to enable API calls. If not set, the system will use mock interpretations for development.

**Note**: Update the `base_url` in `app/services/dedalus_client.py` if your Dedalus API endpoint differs.

## Development Notes

- The system uses mock predictions when models are not available, making it easy to develop and test without trained models
- All audio files are temporarily stored in the `data/` directory during processing
- Logging is configured to show INFO level messages by default
- CORS is enabled for all origins in development (restrict in production)

## Troubleshooting

### Audio Loading Issues

- Ensure `ffmpeg` is installed for MP3/M4A support:
  ```bash
  # Windows (using chocolatey)
  choco install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Linux
  sudo apt-get install ffmpeg
  ```

### Model Loading Issues

- Check that PyTorch models are in the `assets/` directory
- Verify model input/output shapes match expected formats
- Check logs for specific error messages

### Performance

- For production, PyTorch will automatically use GPU if available (CUDA), otherwise CPU
- Adjust segment length and overlap based on your use case
- Use multiple workers for uvicorn in production

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

