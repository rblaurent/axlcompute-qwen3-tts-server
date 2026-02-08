"""Quick benchmark of dffdeeq fork optimizations on CustomVoice model."""
import time
import torch
import numpy as np

torch.set_float32_matmul_precision('high')

from qwen_tts import Qwen3TTSModel

print("Loading model...")
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print(f"Model loaded in {time.time()-t0:.1f}s")

# --- Baseline (no optimizations) ---
text = "Hello, this is a test of the text to speech system. I want to see how fast it can generate audio."
speaker = model.get_supported_speakers()[0]
print(f"Speaker: {speaker}")

print("\n--- Baseline (no optimizations) ---")
for i in range(3):
    t0 = time.time()
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker=speaker)
    elapsed = time.time() - t0
    audio_dur = len(wavs[0]) / sr
    print(f"  Run {i+1}: {elapsed:.2f}s for {audio_dur:.2f}s audio = RTF {elapsed/audio_dur:.2f}")

# --- Enable optimizations ---
print("\nEnabling streaming optimizations...")
model.enable_streaming_optimizations(
    decode_window_frames=300,
    use_compile=True,
    use_cuda_graphs=False,
    compile_mode="max-autotune",
    use_fast_codebook=True,
    compile_codebook_predictor=True,
)

print("\n--- Warmup (compilation) ---")
t0 = time.time()
wavs, sr = model.generate_custom_voice(text="Warmup test.", language="English", speaker=speaker)
print(f"  Warmup: {time.time()-t0:.2f}s")

print("\n--- Optimized ---")
for i in range(3):
    t0 = time.time()
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker=speaker)
    elapsed = time.time() - t0
    audio_dur = len(wavs[0]) / sr
    print(f"  Run {i+1}: {elapsed:.2f}s for {audio_dur:.2f}s audio = RTF {elapsed/audio_dur:.2f}")
