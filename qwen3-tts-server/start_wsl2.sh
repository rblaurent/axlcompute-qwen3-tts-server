#!/bin/bash
# Launch TTS server inside WSL2 with torch.compile optimizations
set -e

source ~/tts-env/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd /mnt/t/Projects/Qwen3-TTS/qwen3-tts-server
python server.py
