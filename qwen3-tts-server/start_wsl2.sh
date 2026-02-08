#!/bin/bash
# Launch TTS server inside WSL2 with torch.compile optimizations
set -e

source ~/tts-env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=~/.triton/cache
# Uncomment for compile debugging:
# export TORCH_COMPILE_DEBUG=1
# export TORCHINDUCTOR_GRAPH_DIAGRAM=1

cd /mnt/t/Projects/Qwen3-TTS/qwen3-tts-server
python server.py
