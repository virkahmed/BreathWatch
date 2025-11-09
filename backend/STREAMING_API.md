# API Documentation

## Architecture Update

**Note: WebSocket streaming is no longer used.** The mobile app processes all audio locally on-device and sends final summaries to the backend.

## Current Endpoints

### POST `/nightly-summary`

**Primary endpoint for mobile app.** Accepts pre-computed summary from mobile app and returns Dedalus AI interpretation.

See [README.md](README.md) for full documentation.

### POST `/analyze`

**Development/Testing Only.** For testing with complete audio files.

See [README.md](README.md) for full documentation.

## Deprecated: WebSocket Streaming

The `/stream` WebSocket endpoint has been removed. The mobile app now:
1. Processes audio locally (recording, preprocessing, feature extraction, ONNX inference)
2. Aggregates detected events
3. Sends final summary to `/nightly-summary` endpoint

This architecture provides:
- Better privacy (audio stays on device)
- Lower latency (no network streaming)
- Reduced bandwidth (only summaries sent)
- Offline capability (works without backend)
