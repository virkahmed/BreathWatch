# Installation Guide

## Quick Install

```bash
cd Backend
pip install -r requirements.txt
```

## Missing Packages

If you encounter import errors, install the missing packages:

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install pydub noisereduce onnxruntime
```

## System Dependencies

### FFmpeg (for audio format support)

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Verify Installation

Run the test script to verify all imports:

```bash
python test_imports.py
```

All imports should show ✓ (checkmark) if installed correctly.

## Troubleshooting

### ImportError: No module named 'pydub'
```bash
pip install pydub
```

### ImportError: No module named 'noisereduce'
```bash
pip install noisereduce
```

### ImportError: No module named 'onnxruntime'
```bash
pip install onnxruntime
```

### Audio format errors
Install FFmpeg (see System Dependencies above).

### Using a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

