# Verification Guide

This guide helps you verify that the backend is working correctly for both UI calls and trained model inference.

## Quick Start

1. **Start the server:**
   ```bash
   python run.py
   ```

2. **In another terminal, run the test script:**
   ```bash
   python test_api.py
   ```

3. **Or manually check status:**
   ```bash
   curl http://localhost:8000/status
   ```

## Verification Checklist

### ✅ 1. Server is Running

**Check:** Open http://localhost:8000 in your browser or run:
```bash
curl http://localhost:8000/
```

**Expected:** `{"status": "healthy", ...}`

### ✅ 2. Models are Loaded

**Check:** GET http://localhost:8000/status

**Expected Response:**
```json
{
  "models": {
    "cough_model": {
      "loaded": true,
      "path": "/path/to/assets/cough_model.onnx",
      "num_outputs": 6,
      "input_shape": [128, 31]
    },
    "wheeze_detector": {
      "loaded": true,
      "path": "/path/to/assets/wheeze_detector.onnx",
      "input_shape": [128, 31]
    }
  }
}
```

**If models are NOT loaded:**
- Models will use mock predictions (works for testing)
- Place models in `assets/` directory:
  - `assets/cough_model.onnx`
  - `assets/wheeze_detector.onnx`
- Restart the server after adding models

### ✅ 3. UI Can Call Endpoints

**Check:** Open http://localhost:8000/docs

**Verify these endpoints exist:**
- `POST /process-chunk` - Process 10-minute audio chunks
- `POST /final-summary/{session_id}` - Get nightly summary
- `POST /nightly-summary` - Legacy endpoint for mobile summaries
- `GET /status` - Check system status

**CORS:** Already configured to allow all origins (for development). In production, update `allow_origins` in `main.py`.

### ✅ 4. Model Inference Works

**Test with actual audio:**

```python
import requests

# Test process-chunk endpoint
with open("test_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process-chunk",
        files={"audio_chunk": f},
        data={
            "chunk_index": 0,
            "session_id": "test-123"
        }
    )
    print(response.json())
```

**Expected:**
- If models loaded: Real predictions with 6 outputs
- If models missing: Mock predictions (still works, but not real)

### ✅ 5. Attribute Probabilities are Checked

**Verify in logs:**
When processing audio, check server logs for:
```
Cough model prediction: cough=0.823, wet=0.456, stridor=0.123
Detected 3 cough events from 600 windows
```

**Verify in response:**
The `/process-chunk` response should include:
- `detected_events` with `tags` array (WET, STRIDOR, etc.)
- Events tagged based on ≥40% threshold

### ✅ 6. Dedalus AI Integration

**Check:** GET http://localhost:8000/status

**Expected:**
```json
{
  "dedalus_ai": {
    "configured": true,
    "base_url": "https://api.dedalus.ai"
  }
}
```

**If not configured:**
- Will use mock interpretations (works for testing)
- Add `DEDALUS_API_KEY=your_key` to `.env` file
- Restart server

## Testing Workflow

### Development (No Models)

1. Start server: `python run.py`
2. Check status: `GET /status` - should show `loaded: false` for models
3. Test endpoints: Use `/process-chunk` - will use mock predictions
4. Verify structure: Responses should have correct schema

### Production (With Models)

1. Place models in `assets/`:
   ```
   assets/
     ├── cough_model.onnx
     └── wheeze_detector.onnx
   ```

2. Start server: `python run.py`
3. Check status: `GET /status` - should show `loaded: true`
4. Test with real audio: Use `/process-chunk` with actual audio file
5. Verify outputs: Check that predictions are realistic (not mock values)

## Common Issues

### Models Not Loading

**Symptoms:** Status shows `loaded: false`

**Solutions:**
1. Check file paths: Models should be in `Backend/assets/`
2. Check file names: Must be exactly `cough_model.onnx` and `wheeze_detector.onnx`
3. Check file permissions: Ensure files are readable
4. Check logs: Look for error messages in server output

### Import Errors

**Symptoms:** `ImportError: cannot import name 'WindowPrediction'`

**Solutions:**
1. Clear Python cache: `Remove-Item -Recurse app\__pycache__`
2. Restart server
3. Verify `schemas.py` has all classes (should be 187 lines)

### CORS Errors (UI)

**Symptoms:** Browser console shows CORS errors

**Solutions:**
1. Check `allow_origins` in `main.py` (currently `["*"]` for dev)
2. In production, specify exact UI origin
3. Ensure server is running on correct port

## End-to-End Test

1. **Start server:**
   ```bash
   python run.py
   ```

2. **Check status:**
   ```bash
   python test_api.py
   ```

3. **Test with audio (if you have a test file):**
   ```bash
   curl -X POST "http://localhost:8000/process-chunk" \
     -F "audio_chunk=@test_audio.wav" \
     -F "chunk_index=0" \
     -F "session_id=test-123"
   ```

4. **Get final summary:**
   ```bash
   curl -X POST "http://localhost:8000/final-summary/test-123" \
     -H "Content-Type: application/json" \
     -d '{"symptom_form": {"fever": false, "duration": 3}}'
   ```

## Next Steps

Once verified:
1. ✅ UI can call endpoints
2. ✅ Models are loaded (or using mocks)
3. ✅ Attribute probabilities are checked
4. ✅ Events are tagged correctly
5. ✅ Dedalus AI is configured (optional)

You're ready to integrate with your frontend!

