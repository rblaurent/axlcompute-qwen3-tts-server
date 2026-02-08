@echo off
REM Qwen3-TTS Server Startup Script
REM Usage: start_server.bat [model_path] [port]

setlocal

REM Configuration
set DEFAULT_PORT=8765
set DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

REM Parse arguments
if not "%~1"=="" (
    set QWEN3_TTS_MODEL=%~1
) else (
    set QWEN3_TTS_MODEL=%DEFAULT_MODEL%
)

if not "%~2"=="" (
    set PORT=%~2
) else (
    set PORT=%DEFAULT_PORT%
)

echo =====================================================
echo   Qwen3-TTS Server
echo =====================================================
echo Model: %QWEN3_TTS_MODEL%
echo Port:  %PORT%
echo Backend: qwen-tts (no true streaming)
echo =====================================================

REM Change to server directory
cd /d "%~dp0"

REM Run the server
python -m uvicorn server:app --host 127.0.0.1 --port %PORT%

endlocal
