"""
Generate streaming audio and save to WAV for quality testing.
"""

import asyncio
import wave
import time
import torch


async def main():
    from qwen_tts import Qwen3TTSModel
    from streaming_generator import StreamingTTSGenerator, StreamingConfig

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    config = StreamingConfig(packet_size=8)
    generator = StreamingTTSGenerator(model, config)

    text = "Hello, this is a test of the streaming text to speech system. The audio should sound smooth without any cuts or artifacts at chunk boundaries."
    print(f"Generating: '{text}'")

    # Collect all audio chunks
    audio_chunks = []
    start_time = time.perf_counter()

    async for audio_bytes in generator.generate_streaming(
        text=text,
        speaker="Serena",
        language="English",
    ):
        audio_chunks.append(audio_bytes)
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"  Packet {len(audio_chunks)}: {len(audio_bytes)} bytes at {elapsed:.0f}ms")

    total_time = (time.perf_counter() - start_time) * 1000
    print(f"\nTotal: {len(audio_chunks)} packets in {total_time:.0f}ms")

    # Save to WAV
    output_file = "streaming_output.wav"
    all_audio = b"".join(audio_chunks)

    with wave.open(output_file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(24000)
        wav.writeframes(all_audio)

    print(f"\nSaved to: {output_file}")
    print(f"Audio length: {len(all_audio) / 2 / 24000:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
