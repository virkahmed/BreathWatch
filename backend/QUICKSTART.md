# Quick Start Guide

## 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: On some systems, you may need to install `ffmpeg` for audio format support:
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

## 2. Run the Server

```bash
python run.py
```

Or:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 3. Test the API

### Option 1: Using the Interactive Docs

Open your browser and navigate to:
- http://localhost:8000/docs

Use the Swagger UI to test the `/analyze` endpoint.

### Option 2: Using curl

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "audio_file=@path/to/your/audio.wav" \
  -F "patient_id=101" \
  -F "age=63" \
  -F "sex=M" \
  -F "sleep_duration_minutes=420"
```

### Option 3: Using Python

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

## 4. Add ML Models (Optional)

For production use, place your trained ONNX models in the `assets/` directory:
- `assets/cough_classifier.onnx`
- `assets/respiratory_classifier.onnx`

**Note**: The system works with mock predictions if models are not provided, which is useful for development.

## 5. Configure Dedalus AI (Optional)

Create a `.env` file in the `backend/` directory:

```env
DEDALUS_API_KEY=your_api_key_here
```

If not configured, the system will use mock interpretations.

## Troubleshooting

- **Import errors**: Make sure you're running from the `backend/` directory or have it in your Python path
- **Audio format errors**: Install `ffmpeg` for MP3/M4A support
- **Model errors**: Check that ONNX models are in the correct format and location

