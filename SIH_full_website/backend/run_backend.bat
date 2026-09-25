@echo off
title AI Voice Deepfake & Scam Detector - Backend Server
echo ========================================================
echo   Starting FastAPI Backend on http://localhost:8000
echo ========================================================
cd /d "%~dp0"
python -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo Installing FastAPI & dependencies...
    python -m pip install fastapi uvicorn python-multipart
)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
