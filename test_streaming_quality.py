#!/usr/bin/env python3
"""
Test streaming audio quality with different parameters.

Generates WAVs for side-by-side comparison:
  - batch (reference quality)
  - streaming with various emit_every_frames and decode_window_frames combos
  - old progressive decode (our streaming_generator approach)

Usage:
  # Run in WSL2:
  python test_streaming_quality.py
"""

import os
import time
import wave

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

TEXT = "Bonjour, je suis ravie de vous accueillir aujourd'hui. Comment puis-je vous aider? C'est un plaisir de discuter avec vous."
SPEAKER = "thalya"
LANGUAGE = "French"
INSTRUCT = "Parle de maniere naturelle et engageante."
SAMPLE_RATE = 24000
OUTPUT_DIR = "test_results/streaming_quality"


def save_wav(audio: np.ndarray, path: str, sr: int = SAMPLE_RATE):
    audio = audio.astype(np.float32)
    mx = max(abs(audio.max()), abs(audio.min()))
    if mx > 1.0:
        audio = audio / mx
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_batch(model):
    """Reference: batch generation."""
    print("\n[batch] Generating...")
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=TEXT, speaker=SPEAKER, language=LANGUAGE, instruct=INSTRUCT,
    )
    elapsed = time.perf_counter() - t0
    audio = wavs[0]
    dur = len(audio) / sr
    print(f"  Time: {elapsed:.2f}s, Audio: {dur:.2f}s, RTF: {elapsed/dur:.2f}x")
    save_wav(audio, os.path.join(OUTPUT_DIR, "00_batch.wav"), sr)
    return audio


def test_streaming(model, emit_every, decode_window, overlap, tag):
    """Test fork streaming with specific parameters."""
    print(f"\n[{tag}] emit_every={emit_every}, decode_window={decode_window}, overlap={overlap}")
    chunks = []
    t0 = time.perf_counter()
    ttfb = None
    for chunk, sr in model.stream_generate_custom_voice(
        text=TEXT, speaker=SPEAKER, language=LANGUAGE, instruct=INSTRUCT,
        emit_every_frames=emit_every,
        decode_window_frames=decode_window,
        overlap_samples=overlap,
    ):
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        chunks.append(chunk)
    elapsed = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    dur = len(audio) / sr if len(audio) > 0 else 0
    print(f"  TTFB: {ttfb*1000:.0f}ms, Time: {elapsed:.2f}s, Audio: {dur:.2f}s, "
          f"RTF: {elapsed/dur:.2f}x, Chunks: {len(chunks)}")
    save_wav(audio, os.path.join(OUTPUT_DIR, f"{tag}.wav"), sr)
    return audio


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        os.path.expanduser("~/models/thalya-checkpoint"),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("Applying optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=80,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="default",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
    )

    # Warmup
    print("Warmup (batch)...")
    model.generate_custom_voice(text="Test.", speaker=SPEAKER, language=LANGUAGE)
    print("Warmup (streaming)...")
    for _ in model.stream_generate_custom_voice(text="Test.", speaker=SPEAKER, language=LANGUAGE):
        pass

    # Reference: batch
    test_batch(model)

    # Streaming variants
    configs = [
        # (emit_every, decode_window, overlap, tag)
        (4,   80,   0,    "01_e4_w80_o0"),
        (8,   80,   0,    "02_e8_w80_o0"),
        (12,  80,   0,    "03_e12_w80_o0"),
        (20,  80,   0,    "04_e20_w80_o0"),
        (4,   80,   512,  "05_e4_w80_o512"),
        (8,   80,   512,  "06_e8_w80_o512"),
        (4,   200,  0,    "07_e4_w200_o0"),
        (8,   200,  0,    "08_e8_w200_o0"),
        (4,   200,  512,  "09_e4_w200_o512"),
        (4,   400,  0,    "10_e4_w400_o0"),
        (8,   400,  0,    "11_e8_w400_o0"),
    ]

    for emit_every, decode_window, overlap, tag in configs:
        try:
            test_streaming(model, emit_every, decode_window, overlap, tag)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Compare WAVs in: {OUTPUT_DIR}/")
    print("  00_batch.wav = reference quality")


if __name__ == "__main__":
    main()
