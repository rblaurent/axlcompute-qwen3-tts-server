"""
Speed comparison test for streaming TTS optimizations.

Compares:
- Old: packet_size=4, subtalker_dosample=True, sync decoding
- New: packet_size=8, subtalker_dosample=False, async decoding
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "qwen3-tts-repo"))

import torch
from streaming_generator import (
    StreamingConfig,
    StreamingTTSGenerator,
    audio_to_pcm_bytes,
)


@dataclass
class TestResult:
    name: str
    first_packet_ms: float
    total_time_ms: float
    num_packets: int
    total_audio_ms: float
    realtime_factor: float  # <1 means faster than realtime


async def test_config(
    model,
    text: str,
    speaker: str,
    language: str,
    config: StreamingConfig,
    use_greedy_subtalker: bool,
    name: str,
) -> TestResult:
    """Test a specific configuration and return metrics."""

    generator = StreamingTTSGenerator(model, config)

    # Override subtalker setting by patching _prepare_generation
    original_prepare = generator._prepare_generation

    def patched_prepare(text, speaker, language, instruct=None):
        result = original_prepare(text, speaker, language, instruct)
        # Override subtalker_dosample
        result['subtalker_dosample'] = not use_greedy_subtalker  # False = greedy
        return result

    if use_greedy_subtalker:
        generator._prepare_generation = patched_prepare
    else:
        # Force sampling for old config
        def force_sampling_prepare(text, speaker, language, instruct=None):
            result = original_prepare(text, speaker, language, instruct)
            result['subtalker_dosample'] = True  # Force sampling
            return result
        generator._prepare_generation = force_sampling_prepare

    start_time = time.perf_counter()
    first_packet_time = None
    total_bytes = 0
    num_packets = 0

    async for audio_bytes in generator.generate_streaming(text, speaker, language):
        if first_packet_time is None:
            first_packet_time = time.perf_counter() - start_time
        total_bytes += len(audio_bytes)
        num_packets += 1

    total_time = time.perf_counter() - start_time

    # Calculate audio duration (16-bit mono @ 24kHz)
    total_samples = total_bytes // 2
    audio_duration_ms = (total_samples / 24000) * 1000

    realtime_factor = (total_time * 1000) / audio_duration_ms if audio_duration_ms > 0 else 0

    return TestResult(
        name=name,
        first_packet_ms=first_packet_time * 1000 if first_packet_time else 0,
        total_time_ms=total_time * 1000,
        num_packets=num_packets,
        total_audio_ms=audio_duration_ms,
        realtime_factor=realtime_factor,
    )


async def run_comparison(model, text: str, speaker: str = "Thalya", language: str = "French"):
    """Run comparison between old and new configurations."""

    print(f"\n{'='*60}")
    print(f"Testing with {len(text)} characters of text")
    print(f"{'='*60}\n")

    # Old configuration
    old_config = StreamingConfig(
        packet_size=4,
        poll_interval=0.01,
    )

    # New configuration
    new_config = StreamingConfig(
        packet_size=8,
        poll_interval=0.005,
    )

    # Warmup run
    print("Warmup run...")
    warmup_config = StreamingConfig(packet_size=4)
    warmup_gen = StreamingTTSGenerator(model, warmup_config)
    async for _ in warmup_gen.generate_streaming("Bonjour.", speaker, language):
        pass
    print("Warmup complete.\n")

    # Test old configuration (with sampling)
    print("Testing OLD config (packet_size=4, subtalker_dosample=True)...")
    old_result = await test_config(
        model, text, speaker, language,
        old_config,
        use_greedy_subtalker=False,
        name="OLD (sampling)",
    )

    # Brief pause between tests
    await asyncio.sleep(1.0)

    # Test new configuration (greedy)
    print("Testing NEW config (packet_size=8, subtalker_dosample=False)...")
    new_result = await test_config(
        model, text, speaker, language,
        new_config,
        use_greedy_subtalker=True,
        name="NEW (greedy)",
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    for result in [old_result, new_result]:
        print(f"{result.name}:")
        print(f"  First packet:    {result.first_packet_ms:>8.0f} ms")
        print(f"  Total time:      {result.total_time_ms:>8.0f} ms")
        print(f"  Audio duration:  {result.total_audio_ms:>8.0f} ms")
        print(f"  Packets:         {result.num_packets:>8}")
        print(f"  Realtime factor: {result.realtime_factor:>8.2f}x (< 1 = faster than playback)")
        print()

    # Comparison
    speedup = old_result.total_time_ms / new_result.total_time_ms if new_result.total_time_ms > 0 else 0
    first_packet_diff = new_result.first_packet_ms - old_result.first_packet_ms

    print(f"{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  Speedup:              {speedup:.2f}x faster")
    print(f"  First packet change:  {first_packet_diff:+.0f} ms")
    print(f"  Old realtime factor:  {old_result.realtime_factor:.2f}x")
    print(f"  New realtime factor:  {new_result.realtime_factor:.2f}x")

    if new_result.realtime_factor < 1.0:
        print(f"\n  [OK] Generation is FASTER than playback!")
    else:
        print(f"\n  [!!] Generation is still slower than playback")
        print(f"    (need {new_result.realtime_factor:.1f}x speedup to match)")


async def main():
    import os
    from qwen_tts import Qwen3TTSModel

    # Use thalya checkpoint
    model_path = os.environ.get(
        "QWEN3_TTS_MODEL",
        "T:/Projects/Qwen3-TTS/thalya/model/checkpoint"
    )

    print(f"Loading model: {model_path}")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Model loaded.\n")

    # Test texts of varying lengths (French)
    short_text = "Bonjour, comment allez-vous aujourd'hui?"

    medium_text = (
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "Ceci est un test du système de synthèse vocale en streaming. "
        "Nous voulons voir à quelle vitesse il peut générer de l'audio."
    )

    long_text = (
        "L'intelligence artificielle a fait des progrès remarquables ces dernières années. "
        "Les grands modèles de langage peuvent désormais comprendre et générer du texte "
        "semblable à celui des humains avec une précision impressionnante. Les systèmes "
        "de synthèse vocale se sont également considérablement améliorés, produisant des "
        "voix naturelles qui sont presque impossibles à distinguer de la parole humaine. "
        "La combinaison de ces technologies permet de nouvelles applications dans "
        "l'accessibilité, la création de contenu et l'interaction homme-machine."
    )

    # Run tests
    await run_comparison(model, short_text)
    await run_comparison(model, medium_text)
    await run_comparison(model, long_text)


if __name__ == "__main__":
    asyncio.run(main())
