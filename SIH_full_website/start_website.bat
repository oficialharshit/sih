@echo off
title voxguard - AI Voice Deepfake & Scam Detection Platform
echo =========================================================================
echo            VOXGUARD - AI VOICE FORENSICS PLATFORM
echo =========================================================================
echo.
echo [1/3] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+
    pause
    exit /b
)

echo.
echo [2/3] Verifying dependencies...
python -c "import fastapi, uvicorn, multipart" 2>nul
if %errorlevel% neq 0 (
    echo Installing required web dependencies...
    python -m pip install fastapi uvicorn python-multipart
)

echo.
echo [3/3] Launching voxguard Server on http://localhost:8000 ...
echo.
cd /d "%~dp0"

:: Open browser after 2 seconds
start "" http://localhost:8000

:: Run using app-dir
python -m uvicorn main:app --app-dir "%~dp0backend" --host 0.0.0.0 --port 8000 --reload

pause
