"""
REAL practical comparison: streaming vs non-streaming.

Measures actual TIME TO FIRST AUDIO for both methods and saves wavs for listening.
"""

import asyncio
import time
import wave
import numpy as np
import torch


def save_pcm_to_wav(pcm_bytes: bytes, filename: str, sample_rate: int = 24000):
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)


def to_pcm(audio: np.ndarray) -> bytes:
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


async def main():
    from qwen_tts import Qwen3TTSModel
    from streaming_generator import StreamingTTSGenerator, StreamingConfig

    print("=" * 60)
    print("REAL COMPARISON: Time to First Audio")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        r"T:\Projects\Qwen3-TTS\thalya\model\checkpoint",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # Test text
    text = "Bonjour, je suis Thalya. Je suis ravie de vous rencontrer aujourd'hui. Comment puis-je vous aider?"
    print(f"\nText ({len(text)} chars): '{text}'")

    # =========================================
    # NON-STREAMING: You wait for EVERYTHING
    # =========================================
    print("\n" + "=" * 60)
    print("NON-STREAMING (batch)")
    print("=" * 60)
    print("Generating... (you wait until 100% complete)")

    start = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker="thalya",
        language="French",
    )
    end = time.perf_counter()

    non_streaming_wait = end - start
    non_streaming_audio = wavs[0]
    non_streaming_pcm = to_pcm(non_streaming_audio)
    non_streaming_duration = len(non_streaming_audio) / 24000

    print(f"\n  TIME TO FIRST AUDIO: {non_streaming_wait*1000:.0f} ms")
    print(f"  (You wait {non_streaming_wait*1000:.0f}ms before hearing anything)")
    print(f"  Audio duration: {non_streaming_duration:.2f}s")

    save_pcm_to_wav(non_streaming_pcm, "compare_non_streaming.wav")
    print(f"  Saved: compare_non_streaming.wav")

    # =========================================
    # STREAMING: You get audio incrementally
    # =========================================
    print("\n" + "=" * 60)
    print("STREAMING (progressive)")
    print("=" * 60)

    # Use realistic pre-buffering (auto-calculated from text length)
    from streaming_generator import estimate_required_buffer_ms
    estimated_buffer = estimate_required_buffer_ms(text)

    config = StreamingConfig(
        packet_size=8,
        pre_buffer_ms=0,  # 0 = auto-calculate based on text length
    )
    generator = StreamingTTSGenerator(model, config)

    print(f"Estimated pre-buffer needed: {estimated_buffer} ms")
    print("Generating... (audio arrives in chunks, after pre-buffer fills)")

    streaming_chunks = []
    start = time.perf_counter()
    first_packet_time = None

    async for audio_bytes in generator.generate_streaming(
        text=text,
        speaker="thalya",
        language="French",
    ):
        now = time.perf_counter()
        if first_packet_time is None:
            first_packet_time = now - start
            print(f"\n  >> FIRST AUDIO ARRIVED at {first_packet_time*1000:.0f} ms <<")
        streaming_chunks.append(audio_bytes)

    end = time.perf_counter()
    streaming_total = end - start

    streaming_pcm = b''.join(streaming_chunks)
    streaming_duration = (len(streaming_pcm) // 2) / 24000

    print(f"\n  TIME TO FIRST AUDIO: {first_packet_time*1000:.0f} ms")
    print(f"  Total generation time: {streaming_total*1000:.0f} ms")
    print(f"  Audio duration: {streaming_duration:.2f}s")

    save_pcm_to_wav(streaming_pcm, "compare_streaming.wav")
    print(f"  Saved: compare_streaming.wav")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speedup = non_streaming_wait / first_packet_time if first_packet_time > 0 else 0

    print(f"""
  Non-streaming wait: {non_streaming_wait*1000:>6.0f} ms  (wait for 100% before any audio)
  Streaming wait:     {first_packet_time*1000:>6.0f} ms  (first chunk arrives)

  SPEEDUP: {speedup:.1f}x faster to first audio with streaming!
  You save {(non_streaming_wait - first_packet_time)*1000:.0f} ms of waiting.

  Files to compare (listen to both):
    1. compare_non_streaming.wav
    2. compare_streaming.wav

  They should sound similar (different generation runs, but same quality).
""")

    return speedup > 1.0


if __name__ == "__main__":
    success = asyncio.run(main())
    print("=" * 60)
    print("RESULT:", "STREAMING IS FASTER" if success else "NO IMPROVEMENT")
    print("=" * 60)
