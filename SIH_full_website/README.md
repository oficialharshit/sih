# 🎙️ voxguard - AI Voice Deepfake & Scam Detection Platform

A full-stack, multimodal artificial intelligence platform designed to detect synthetic AI voice cloning, transcribe speech in real-time, and analyze conversations for fraud and scam patterns.

---

## 🌟 Key Features

1. **🎙️ Voice Deepfake Classification**:
   - Audio preprocessing & Voice Activity Detection (VAD).
   - Wav2Vec2 transformer acoustic feature analysis.
   - Real-time confidence metrics for **Real Human Voice** vs. **AI-Generated Clone**.

2. **📝 Automatic Speech Recognition (ASR)**:
   - Powered by `openai/whisper-small` for accurate multilingual speech-to-text.

3. **🚨 NLP Scam & Social Engineering Classifier**:
   - Fine-tuned **DistilBERT** classification engine.
   - Identifies high-risk fraud keywords (e.g. banking impersonation, OTP requests, urgency tactics).

4. **📊 Weighted Risk Engine**:
   - Unified Threat Index: **70% Acoustic Deepfake Score + 30% NLP Scam Score**.
   - Dynamic classification: `LOW RISK (🛡️)`, `MEDIUM RISK (⚠️)`, `HIGH RISK (🚨)`.

5. **⚡ Full-Stack Web Application**:
   - **Frontend**: High-tech cybersecurity dark UI with live Web Audio API microphone recording, drag-and-drop audio player, and chunk-by-chunk confidence visualizer.
   - **Backend**: High-performance FastAPI server with RESTful endpoints and GPU (CUDA) acceleration support.

---

## 🚀 Quick Start (1-Click Run)

### Prerequisites:
- Python 3.10 or higher
- NVIDIA GPU with CUDA (recommended, but runs on CPU automatically as well)

### Launching the Website:
Simply double-click:
```bash
start_website.bat
```
or open your terminal and run:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📁 Project Structure

```
SIH_full_website/
├── backend/
│   ├── analysis/            # Aggregation, NLP predictor, risk engine, pipeline
│   ├── audio/               # Audio loader, chunking, VAD, preprocessing
│   ├── model/               # Wav2Vec2 audio deepfake detector
│   ├── scam_distilbert/     # Fine-tuned DistilBERT weights & tokenizer
│   ├── speech/              # Whisper STT transcriber
│   ├── static/              # Web application UI (HTML5, Tailwind, Lucide)
│   ├── test_audio/          # Sample audio clips for instant testing
│   ├── main.py              # FastAPI server & endpoints
│   ├── requirements.txt     # Python backend dependencies
│   └── run_backend.bat      # Windows backend starter
├── frontend/
│   ├── public/              # Static assets and template
│   ├── src/                 # Source components (React)
│   └── package.json         # Optional React/Vite development setup
├── start_website.bat        # 1-click full platform launcher
└── README.md                # Documentation & Architecture guide
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and GPU telemetry |
| `GET` | `/api/sample-audio` | List available demo audio clips |
| `GET` | `/api/sample-audio/{filename}` | Serve a demo audio file |
| `POST` | `/api/analyze` | Multipart audio file upload -> complete forensic JSON response |
