🎤 BreathWatch – AI-Powered Sleep Respiratory Monitor

Hi everyone, we’re excited to introduce BreathWatch — a smart respiratory monitoring app that helps people track and understand their nighttime breathing health using AI-powered audio analysis.

🌙 The Problem

Millions of people suffer from undiagnosed cough, asthma, and sleep-related breathing disorders. Traditional monitoring requires bulky medical devices or hospital visits.
We wanted to make respiratory health tracking as easy as pressing record on your phone.

💡 Our Solution: BreathWatch

BreathWatch uses your phone’s microphone to record audio overnight, detect coughs and wheezes in real time, and generate a morning health summary — all automatically.

Here’s how it works:

The frontend is built with React Native and Expo, so it runs seamlessly on both mobile and web.
It records audio using expo-av or the Web Audio API, converts it into clean 16 kHz WAV chunks, and uploads them live.

The backend, built with FastAPI and PyTorch, receives each chunk, cleans the sound with librosa and noisereduce, and runs our CoughMultitaskCNN and WheezeDetector models.
It identifies coughs, classifies their type — like wet, stridor, or congestion — and detects wheezing episodes.

When recording ends, the backend aggregates everything into a Nightly Summary:

Coughs per hour

Wheeze time percentage

Bout lengths and frequency

Attribute breakdowns

Optional AI interpretation using Dedalus for health insights

📊 The Experience

On the dashboard, users see real-time cough counts while recording.
In the morning, they get visual analytics — bar charts, pattern scores, and AI explanations — that help them understand trends like “possible asthma-like patterns” or “COPD-like tendencies.”

⚙️ Tech Stack Summary

Frontend: Expo + React Native, MUI Charts, TypeScript

Backend: FastAPI, Librosa, PyTorch, Dedalus AI

AI Models: CoughMultitaskCNN + WheezeDetector

Everything runs locally — no special hardware, just your phone.

🚀 Impact

BreathWatch transforms ordinary sleep into actionable respiratory data, making early detection of chronic conditions accessible, affordable, and private.

In short — we’re bringing clinical-grade respiratory insight to everyone, right from their pillow.
