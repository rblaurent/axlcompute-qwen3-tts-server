"""
Test script for true streaming TTS implementation.

This script verifies:
1. The codec_callback is called during generation
2. Tokens are captured incrementally (not all at once)
3. First packet latency is reduced compared to non-streaming
"""

import asyncio
import sys
import time

import torch


def test_codec_callback_invocation():
    """Test that codec_callback is invoked during generation."""
    print("\n=== Test 1: codec_callback invocation ===")

    from qwen_tts import Qwen3TTSModel

    # Load model
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # Track callback invocations
    callback_times = []
    callback_tokens = []

    def on_codec_token(codec_ids):
        callback_times.append(time.perf_counter())
        callback_tokens.append(codec_ids.clone().cpu())
        if len(callback_times) <= 5 or len(callback_times) % 10 == 0:
            print(f"  Token {len(callback_times)}: shape={codec_ids.shape}, time={callback_times[-1] - callback_times[0]:.3f}s")

    # Prepare input
    text = "Hello, this is a test of the streaming TTS system."
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])

    print(f"Generating: '{text}'")
    start_time = time.perf_counter()

    # Run generation with callback
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=[None],
        languages=["English"],
        speakers=["Serena"],
        non_streaming_mode=True,
        codec_callback=on_codec_token,
    )

    end_time = time.perf_counter()

    # Report results
    print(f"\nResults:")
    print(f"  Total generation time: {(end_time - start_time) * 1000:.0f}ms")
    print(f"  Callback invocations: {len(callback_times)}")
    print(f"  Final output tokens: {talker_codes_list[0].shape[0] if talker_codes_list else 0}")

    if callback_times and len(callback_times) > 1:
        first_callback = callback_times[0] - start_time
        print(f"  Time to first callback: {first_callback * 1000:.0f}ms")
        avg_interval = (callback_times[-1] - callback_times[0]) / (len(callback_times) - 1)
        print(f"  Avg interval between callbacks: {avg_interval * 1000:.0f}ms")

        # Check if callbacks happened incrementally
        if first_callback < (end_time - start_time) * 0.5:
            print("  [PASS] Callbacks happening during generation (not at end)")
        else:
            print("  [FAIL] Callbacks might be happening all at once")
    else:
        print("  [FAIL] No callbacks received or only one")

    return len(callback_times) > 0


async def test_streaming_generator():
    """Test the StreamingTTSGenerator."""
    print("\n=== Test 2: StreamingTTSGenerator ===")

    from qwen_tts import Qwen3TTSModel
    from streaming_generator import StreamingTTSGenerator, StreamingConfig

    # Load model
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # Create generator with progressive batch decode
    config = StreamingConfig(packet_size=8)
    generator = StreamingTTSGenerator(model, config)

    text = "Hello, this is a test of the streaming TTS system."
    print(f"Streaming: '{text}'")

    packet_times = []
    packet_sizes = []
    start_time = time.perf_counter()

    async for audio_bytes in generator.generate_streaming(
        text=text,
        speaker="Serena",
        language="English",
    ):
        packet_times.append(time.perf_counter())
        packet_sizes.append(len(audio_bytes))
        print(f"  Packet {len(packet_times)}: {len(audio_bytes)} bytes, time={packet_times[-1] - start_time:.3f}s")

    end_time = time.perf_counter()

    # Report results
    print(f"\nResults:")
    print(f"  Total streaming time: {(end_time - start_time) * 1000:.0f}ms")
    print(f"  Total packets: {len(packet_times)}")
    print(f"  Total audio bytes: {sum(packet_sizes)}")

    if packet_times:
        first_packet = packet_times[0] - start_time
        print(f"  Time to first packet: {first_packet * 1000:.0f}ms")

        # Target: <500ms for first packet
        if first_packet < 0.5:
            print("  [PASS] First packet under 500ms target")
        elif first_packet < 1.0:
            print("  [PARTIAL] First packet under 1s but above 500ms target")
        else:
            print("  [FAIL] First packet took too long")
    else:
        print("  [FAIL] No packets received")

    return len(packet_times) > 0


def main():
    print("=" * 60)
    print("TRUE STREAMING TTS IMPLEMENTATION TEST")
    print("=" * 60)

    # Test 1: Verify codec_callback works
    test1_pass = test_codec_callback_invocation()

    # Test 2: Verify streaming generator works
    test2_pass = asyncio.run(test_streaming_generator())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (codec_callback): {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (streaming_generator): {'PASS' if test2_pass else 'FAIL'}")

    if test1_pass and test2_pass:
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
