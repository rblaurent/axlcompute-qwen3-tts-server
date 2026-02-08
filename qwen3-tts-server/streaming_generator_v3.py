"""
Sentence-Level Streaming TTS Generator for Qwen3-TTS (v3)

Since true token-level streaming requires significant model modifications,
this module implements a more practical approach:

1. Split input text into sentences
2. Generate each sentence as it's requested
3. Stream audio packets for each sentence as soon as it's generated

This gives "sentence-level streaming" - first audio arrives after the first
sentence is generated (~1-2s for short sentences), rather than waiting for
the entire text (~5-10s for paragraphs).

For short sentences, this can achieve <500ms first-packet latency.
"""

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    packet_size: int = 8  # Tokens per audio packet for decode
    left_context: int = 25  # Context for smooth audio boundaries
    sample_rate: int = 24000
    min_sentence_length: int = 5  # Minimum chars for a sentence


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for incremental generation."""
    # Split on sentence-ending punctuation
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

    # If no sentences found, return the whole text
    if not sentences:
        return [text.strip()] if text.strip() else []

    return sentences


class SentenceStreamingGenerator:
    """
    Sentence-level streaming for Qwen3-TTS.

    Generates audio sentence-by-sentence and streams packets
    as each sentence completes.
    """

    def __init__(self, model, config: Optional[StreamingConfig] = None):
        self.model = model
        self.config = config or StreamingConfig()

        # Get decoder reference
        self.decoder = model.model.speech_tokenizer.model.decoder
        self.upsample_rate = self.decoder.total_upsample

    def _generate_sentence(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Generate audio for a single sentence."""
        wavs, sample_rate = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )
        return wavs[0], sample_rate

    def _chunk_audio(self, audio: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
        """Split audio into chunks for streaming."""
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

    def _audio_to_pcm(self, audio: np.ndarray) -> bytes:
        """Convert audio array to PCM bytes."""
        audio = np.clip(audio.astype(np.float32), -1, 1)
        return (audio * 32767).astype(np.int16).tobytes()

    async def generate_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio with sentence-level streaming.

        Splits text into sentences and yields audio packets
        as each sentence is generated.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return

        print(f"[SentenceStream] Split into {len(sentences)} sentences")

        start_time = time.perf_counter()
        first_packet_time = None
        total_packets = 0

        # Calculate chunk size (samples per packet)
        # ~320ms per packet at 24kHz = 7680 samples
        chunk_samples = int(self.config.sample_rate * 0.32)

        for i, sentence in enumerate(sentences):
            sentence_start = time.perf_counter()

            # Generate audio for this sentence
            # Run in thread to not block event loop
            audio, sample_rate = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=sentence: self._generate_sentence(
                    s, speaker, language, instruct
                )
            )

            gen_time = time.perf_counter() - sentence_start

            # Split into chunks and yield
            chunks = self._chunk_audio(audio, chunk_samples)

            for chunk in chunks:
                if first_packet_time is None:
                    first_packet_time = time.perf_counter() - start_time
                    print(f"[SentenceStream] First packet at {first_packet_time*1000:.0f}ms")

                total_packets += 1
                yield self._audio_to_pcm(chunk)

                # Small delay to simulate streaming
                await asyncio.sleep(0.001)

            print(f"[SentenceStream] Sentence {i+1}/{len(sentences)}: "
                  f"gen={gen_time*1000:.0f}ms, chunks={len(chunks)}")

        total_time = time.perf_counter() - start_time
        print(f"[SentenceStream] Complete: {total_packets} packets in {total_time*1000:.0f}ms")


class ParallelSentenceStreamingGenerator:
    """
    Parallel sentence-level streaming for better latency.

    Starts generating subsequent sentences while streaming the current one.
    This can significantly reduce perceived latency for longer texts.
    """

    def __init__(self, model, config: Optional[StreamingConfig] = None):
        self.model = model
        self.config = config or StreamingConfig()

        self.decoder = model.model.speech_tokenizer.model.decoder
        self.upsample_rate = self.decoder.total_upsample

    def _generate_sentence_sync(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> np.ndarray:
        """Generate audio for a sentence (synchronous)."""
        wavs, _ = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )
        return wavs[0]

    def _audio_to_pcm(self, audio: np.ndarray) -> bytes:
        """Convert audio array to PCM bytes."""
        audio = np.clip(audio.astype(np.float32), -1, 1)
        return (audio * 32767).astype(np.int16).tobytes()

    async def generate_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate with parallel sentence processing.

        While streaming one sentence, start generating the next.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return

        print(f"[ParallelStream] Split into {len(sentences)} sentences")

        start_time = time.perf_counter()
        first_packet_time = None
        total_packets = 0
        chunk_samples = int(self.config.sample_rate * 0.32)

        # Queue for generated audio
        audio_queue: Queue = Queue()
        generation_complete = threading.Event()
        generation_error: List[Exception] = []

        def generate_all_sentences():
            """Background thread to generate all sentences."""
            try:
                for i, sentence in enumerate(sentences):
                    audio = self._generate_sentence_sync(
                        sentence, speaker, language, instruct
                    )
                    audio_queue.put((i, audio))
            except Exception as e:
                generation_error.append(e)
            finally:
                generation_complete.set()

        # Start background generation
        gen_thread = threading.Thread(target=generate_all_sentences, daemon=True)
        gen_thread.start()

        # Stream audio as it becomes available
        next_sentence_idx = 0
        pending_audio = {}

        while True:
            # Check for errors
            if generation_error:
                raise generation_error[0]

            # Try to get newly generated audio
            try:
                idx, audio = audio_queue.get_nowait()
                pending_audio[idx] = audio
            except Empty:
                pass

            # Yield audio in order
            while next_sentence_idx in pending_audio:
                audio = pending_audio.pop(next_sentence_idx)

                # Chunk and yield
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i + chunk_samples]
                    if len(chunk) > 0:
                        if first_packet_time is None:
                            first_packet_time = time.perf_counter() - start_time
                            print(f"[ParallelStream] First packet at {first_packet_time*1000:.0f}ms")

                        total_packets += 1
                        yield self._audio_to_pcm(chunk)

                next_sentence_idx += 1

            # Check if done
            if generation_complete.is_set() and next_sentence_idx >= len(sentences):
                break

            # Wait a bit for more audio
            await asyncio.sleep(0.01)

        gen_thread.join(timeout=1.0)

        total_time = time.perf_counter() - start_time
        print(f"[ParallelStream] Complete: {total_packets} packets in {total_time*1000:.0f}ms")


# Export the best generator
StreamingTTSGenerator = SentenceStreamingGenerator
