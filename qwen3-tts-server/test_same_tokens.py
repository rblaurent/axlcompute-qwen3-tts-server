"""
Compare streaming vs non-streaming using the SAME codec tokens.

This captures tokens from one generation and decodes them both ways
to verify streaming produces identical audio to batch decoding.
"""

import asyncio
import time
import wave
import numpy as np
import torch


def save_pcm_to_wav(pcm_bytes: bytes, filename: str, sample_rate: int = 24000):
    """Save raw PCM bytes (s16le) to a WAV file."""
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    print(f"Saved: {filename}")


def to_pcm(audio: np.ndarray) -> bytes:
    """Convert float audio to PCM bytes."""
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


async def test_same_tokens():
    """Compare streaming vs batch decode using identical tokens."""
    print("\n" + "=" * 60)
    print("SAME-TOKEN COMPARISON: Streaming vs Batch")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel
    from streaming_generator import StreamingTTSGenerator, StreamingConfig

    # Load model
    checkpoint_path = r"T:\Projects\Qwen3-TTS\thalya\model\checkpoint"
    print(f"\nLoading model from: {checkpoint_path}")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    text = "Bonjour, je suis Thalya. Comment allez-vous aujourd'hui?"
    print(f"Text: '{text}'")

    # Get decoder reference
    decoder = model.model.speech_tokenizer.model.decoder

    # =====================
    # Method 1: Streaming generation (captures tokens internally)
    # =====================
    print("\n--- Method 1: Streaming generation ---")

    config = StreamingConfig(packet_size=8, pre_buffer_ms=100)  # Short buffer for testing
    generator = StreamingTTSGenerator(model, config)

    # Collect streaming audio
    streaming_chunks = []
    captured_tokens = []

    # We need to capture tokens during streaming
    # Patch the token callback to also save tokens
    original_run = generator._run_generation

    def patched_run(gen_kwargs, token_buffer):
        original_callback = None

        def capturing_callback(codec_ids):
            captured_tokens.append(codec_ids.squeeze(0).cpu().clone())
            token_buffer.add_token(codec_ids.squeeze(0))

        # Replace the callback setup
        try:
            def on_codec_token(codec_ids):
                capturing_callback(codec_ids)

            model.model.generate(
                codec_callback=on_codec_token,
                **gen_kwargs,
            )
            token_buffer.mark_complete()
        except Exception as e:
            token_buffer.mark_error(e)

    generator._run_generation = patched_run

    start_time = time.perf_counter()
    async for audio_bytes in generator.generate_streaming(
        text=text,
        speaker="thalya",
        language="French",
    ):
        streaming_chunks.append(audio_bytes)

    streaming_time = time.perf_counter() - start_time

    # Combine streaming audio
    streaming_pcm = b''.join(streaming_chunks)
    streaming_samples = len(streaming_pcm) // 2
    print(f"Streaming: {streaming_samples} samples ({streaming_samples/24000:.2f}s) in {streaming_time*1000:.0f}ms")
    print(f"Captured {len(captured_tokens)} codec tokens")

    # =====================
    # Method 2: Batch decode the SAME tokens
    # =====================
    print("\n--- Method 2: Batch decode (same tokens) ---")

    if not captured_tokens:
        print("ERROR: No tokens captured!")
        return False

    # Stack tokens: list of (num_codebooks,) -> (seq_len, num_codebooks)
    codec_tokens = torch.stack(captured_tokens, dim=0)
    print(f"Token shape: {codec_tokens.shape}")

    # Decode all at once
    codes = codec_tokens.T.unsqueeze(0).to(decoder.device)

    start_time = time.perf_counter()
    with torch.no_grad():
        wav_batch = decoder(codes)
    batch_time = time.perf_counter() - start_time

    batch_audio = wav_batch.squeeze().float().cpu().numpy()
    batch_pcm = to_pcm(batch_audio)
    print(f"Batch: {len(batch_audio)} samples ({len(batch_audio)/24000:.2f}s) in {batch_time*1000:.0f}ms")

    # =====================
    # Compare
    # =====================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Convert streaming PCM back to float for comparison
    streaming_int16 = np.frombuffer(streaming_pcm, dtype=np.int16)
    streaming_audio = streaming_int16.astype(np.float32) / 32767.0

    print(f"\nLength:")
    print(f"  Batch: {len(batch_audio)} samples")
    print(f"  Streaming: {len(streaming_audio)} samples")
    print(f"  Difference: {abs(len(batch_audio) - len(streaming_audio))} samples")

    # Compare overlapping portion
    min_len = min(len(batch_audio), len(streaming_audio))
    batch_trim = batch_audio[:min_len]
    stream_trim = streaming_audio[:min_len]

    diff = np.abs(batch_trim - stream_trim)

    print(f"\nDifference metrics:")
    print(f"  Max: {diff.max():.6f} ({diff.max()*100:.2f}%)")
    print(f"  Mean: {diff.mean():.6f} ({diff.mean()*100:.3f}%)")
    print(f"  RMS: {np.sqrt((diff**2).mean()):.6f}")

    # Check quality
    max_ok = diff.max() < 0.05  # 5% max
    mean_ok = diff.mean() < 0.01  # 1% mean

    if max_ok and mean_ok:
        print(f"\n[PASS]: Streaming matches batch decode (perceptually identical)")
    else:
        print(f"\n[FAIL]: Significant differences detected")

    # Save files
    print("\n--- Saving files ---")
    save_pcm_to_wav(batch_pcm, "test_same_tokens_batch.wav")
    save_pcm_to_wav(streaming_pcm, "test_same_tokens_streaming.wav")

    print("\nCompare these files - they should sound IDENTICAL:")
    print("  1. test_same_tokens_batch.wav     - Batch decode")
    print("  2. test_same_tokens_streaming.wav - Streaming decode")

    return max_ok and mean_ok


if __name__ == "__main__":
    success = asyncio.run(test_same_tokens())
    print("\n" + "=" * 60)
    print("RESULT:", "PASS" if success else "FAIL")
    print("=" * 60)
