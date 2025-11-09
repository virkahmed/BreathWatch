# Quick Start Guide - Testing Your API

## Step 1: Start the Server

Open a terminal in the `Backend` directory and run:

```bash
python run.py
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
```

The server is now running at `http://localhost:8000`

## Step 2: Run the Test Suite

**In a NEW terminal window** (keep the server running), run:

```bash
cd Backend
python test_api_comprehensive.py
```

This will test:
- ✅ Health check endpoint
- ✅ Status endpoint (model loading)
- ✅ API documentation
- ✅ Process chunk endpoint
- ✅ Final summary endpoint

## Step 3: Manual Testing

### Option A: Use the Browser

1. **Health Check**: http://localhost:8000/
2. **Status**: http://localhost:8000/status
3. **API Docs**: http://localhost:8000/docs (if OpenAPI works)

### Option B: Use curl (Command Line)

```bash
# Health check
curl http://localhost:8000/

# Status
curl http://localhost:8000/status

# Process chunk (requires audio file)
curl -X POST "http://localhost:8000/process-chunk" \
  -F "audio_chunk=@your_audio.wav" \
  -F "chunk_index=0" \
  -F "session_id=test123"
```

### Option C: Use Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Status
response = requests.get("http://localhost:8000/status")
print(response.json())
```

## What to Check

### ✅ Server is Running
- No errors in terminal
- Can access http://localhost:8000/

### ✅ Models are Loaded
Check `/status` endpoint:
```json
{
  "models": {
    "cough_model": {"loaded": true},
    "wheeze_detector": {"loaded": true}
  }
}
```

If `loaded: false`, check:
- `app/models/cough_multitask.pt` exists
- `app/models/wheeze_head.pt` exists

### ✅ Endpoints Work
- `/` returns health status
- `/status` returns model status
- `/process-chunk` accepts POST requests
- `/final-summary/{session_id}` accepts POST requests

## Common Issues

### "Connection refused"
- Server is not running
- Run `python run.py` first

### "Models not loaded"
- Check that `.pt` files exist in `app/models/`
- Check server logs for loading errors

### "404 Not Found"
- Check the exact endpoint path
- Make sure server is running
- Check for typos in the URL

### "500 Internal Server Error"
- Check server logs for detailed error
- Verify model files are valid PyTorch models
- Check that all dependencies are installed

## Next Steps

Once your API is working:
1. Test with actual audio files from your React Native app
2. Verify model predictions are reasonable
3. Test the full flow: chunk processing → final summary
4. Integrate with your frontend

