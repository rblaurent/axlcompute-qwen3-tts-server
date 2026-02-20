#!/bin/bash
# RunPod startup script for Qwen3-TTS server.
#
# First boot (~3-5 min): installs deps to /workspace/venv, downloads base model.
# Subsequent boots (~30s): activates venv, starts server immediately.
#
# Setup:
#   1. Create a RunPod pod with template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   2. Set Volume Disk >= 20GB, mounted at /workspace
#   3. SSH into the pod and copy server files:
#        mkdir -p /workspace/server
#        scp server.py emotion_mapper.py streaming_generator.py /workspace/server/
#        scp runpod_start.sh /workspace/server/
#   4. Run first-time setup:  bash /workspace/server/runpod_start.sh
#   5. Stop the pod. Set Start Command to:  bash /workspace/server/runpod_start.sh
#   6. The relay will auto-start/stop the pod from here.

set -e

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/venv
SERVER_DIR=$WORKSPACE/server
SETUP_MARKER=$WORKSPACE/.setup_complete
MODEL_NAME="${QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
PORT="${PORT:-8765}"

# Persist HuggingFace cache on volume so models survive restarts
export HF_HOME=$WORKSPACE/hf_cache

echo "=== Qwen3-TTS RunPod Startup ==="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"

# --- First-time setup ---
if [ ! -f "$SETUP_MARKER" ]; then
    echo ""
    echo "=== First-time setup (this takes ~3-5 minutes) ==="

    # Create virtual environment on the volume
    echo "[1/4] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # Install dependencies
    echo "[2/4] Installing Python dependencies..."
    pip install --no-cache-dir \
        "qwen-tts>=0.0.5" \
        "fastapi>=0.100.0" \
        "uvicorn[standard]>=0.22.0" \
        "numpy>=1.24.0" \
        "soundfile>=0.13.0" \
        "torch>=2.1.0"

    # Install flash-attn (compiles from source, needs CUDA devel headers from base image)
    echo "[3/4] Installing flash-attn (this takes ~2 min to compile)..."
    pip install flash-attn --no-build-isolation

    # Pre-download the base model
    echo "[4/4] Downloading base model: $MODEL_NAME..."
    python -c "
from qwen_tts import Qwen3TTSModel
Qwen3TTSModel.from_pretrained('$MODEL_NAME')
print('Model downloaded successfully')
"

    touch "$SETUP_MARKER"
    echo ""
    echo "=== Setup complete! ==="
    echo ""
fi

# --- Start server ---
source "$VENV_DIR/bin/activate"

cd "$SERVER_DIR"
echo "Starting server on 0.0.0.0:$PORT..."
exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"
