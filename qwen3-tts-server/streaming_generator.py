"""
Streaming TTS Generator for Qwen3-TTS

This module provides true token-level streaming for Qwen3-TTS by:
1. Running generation in a background thread
2. Capturing codec tokens as they're generated
3. Decoding audio packets incrementally using progressive batch decoding
4. Yielding audio bytes via async generator

Key insight: Progressive batch decoding always decodes ALL tokens accumulated so far,
outputting only the NEW audio portion each time. This guarantees identical quality
to batch decoding since each decode has full left context.
"""

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger("qwen3_tts.streaming")


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    packet_size: int = 8  # Tokens per packet (8 = ~640ms audio, larger = fewer decodes)
    sample_rate: int = 24000
    poll_interval: float = 0.01  # Seconds between queue polls
    pre_buffer_ms: int = 0  # 0 = auto-calculate from text length


@dataclass
class StreamingStats:
    """Statistics from streaming generation."""
    first_packet_time_ms: float = 0.0
    total_packets: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    decode_time_ms: float = 0.0


class ProgressiveTokenBuffer:
    """
    Thread-safe buffer for progressive batch decoding.

    Collects tokens during generation and tracks how many have been
    decoded to audio. Progressive decoding always decodes ALL tokens
    but only outputs the NEW audio portion.
    """

    def __init__(self, packet_size: int = 8):
        self.packet_size = packet_size
        self._tokens: List[torch.Tensor] = []
        self._decoded_token_count: int = 0  # Tokens already decoded
        self._output_sample_count: int = 0  # Audio samples already output
        self._lock = threading.Lock()
        self._generation_complete = threading.Event()
        self._error: Optional[Exception] = None

    def add_token(self, token: torch.Tensor):
        """Add a single codec token (thread-safe)."""
        with self._lock:
            self._tokens.append(token.cpu().clone())

    def add_tokens(self, tokens: torch.Tensor):
        """Add multiple codec tokens at once (thread-safe)."""
        with self._lock:
            # tokens shape: (seq_len, num_codebooks) or (num_codebooks,)
            if tokens.dim() == 1:
                self._tokens.append(tokens.cpu().clone())
            else:
                for t in tokens:
                    self._tokens.append(t.cpu().clone())

    def has_new_packet(self) -> bool:
        """Check if we have enough NEW tokens to warrant decoding."""
        with self._lock:
            new_tokens = len(self._tokens) - self._decoded_token_count
            return new_tokens >= self.packet_size

    def get_all_tokens(self) -> Tuple[Optional[torch.Tensor], int, int]:
        """
        Get all tokens and tracking info for progressive decode.

        Returns:
            Tuple of (codes tensor, decoded_token_count, output_sample_count)
            codes tensor shape: (1, num_codebooks, seq_len)
        """
        with self._lock:
            if not self._tokens:
                return None, 0, 0

            # Stack: list of (num_codebooks,) -> (seq_len, num_codebooks)
            # Then transpose to (num_codebooks, seq_len) and add batch dim
            stacked = torch.stack(self._tokens, dim=0)
            codes = stacked.T.unsqueeze(0)  # (1, num_codebooks, seq_len)

            return codes, self._decoded_token_count, self._output_sample_count

    def mark_decoded(self, token_count: int, sample_count: int):
        """Mark tokens as decoded and track samples output."""
        with self._lock:
            self._decoded_token_count = token_count
            self._output_sample_count = sample_count

    @property
    def token_count(self) -> int:
        """Current total number of tokens."""
        with self._lock:
            return len(self._tokens)

    @property
    def decoded_token_count(self) -> int:
        """Number of tokens already decoded to audio."""
        with self._lock:
            return self._decoded_token_count

    @property
    def output_sample_count(self) -> int:
        """Number of audio samples already output."""
        with self._lock:
            return self._output_sample_count

    def mark_complete(self):
        """Signal that generation is complete."""
        self._generation_complete.set()

    def mark_error(self, error: Exception):
        """Signal an error occurred."""
        self._error = error
        self._generation_complete.set()

    @property
    def is_complete(self) -> bool:
        return self._generation_complete.is_set()

    @property
    def error(self) -> Optional[Exception]:
        return self._error


def audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert audio array to PCM bytes (s16le)."""
    # Ensure float32
    audio = audio.astype(np.float32)

    # Clip to [-1, 1] instead of normalizing per-chunk
    # Per-chunk normalization causes amplitude jumps between chunks
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def estimate_required_buffer_ms(text: str, realtime_factor: float = 2.0) -> int:
    """
    Estimate required pre-buffer from text length.

    Generation is ~1.5-2x slower than real-time, so we need enough buffer
    to cover the gap between generation and playback speeds.

    Args:
        text: Input text to synthesize
        realtime_factor: Ratio of generation time to audio duration (default 2.0)

    Returns:
        Required buffer in milliseconds
    """
    # Estimate audio duration from text
    chars_per_second = 12
    estimated_audio_ms = len(text) * 1000 / chars_per_second

    # Buffer enough to cover generation lag
    # For 1.5x real-time generation: need ~33% of audio as buffer
    # For 2x real-time generation: need ~50% of audio as buffer
    # Use conservative estimate (2x) and cap at reasonable limits
    buffer_ratio = 0.5  # 50% of estimated audio duration
    required_buffer_ms = int(estimated_audio_ms * buffer_ratio)

    # Minimum 1.5s buffer for smooth playback, max 5s to avoid long waits
    return max(1500, min(required_buffer_ms, 5000))


class StreamingTTSGenerator:
    """
    Token-level streaming generator for Qwen3-TTS using progressive batch decoding.

    This class manages the streaming generation pipeline:
    1. Prepares inputs and starts generation in background thread
    2. Monitors token buffer and decodes ALL tokens when enough NEW ones accumulate
    3. Outputs only the NEW audio portion (audio for newly-decoded tokens)
    4. Yields audio bytes as they become available

    Progressive batch decoding ensures identical quality to non-streaming mode
    because each decode has full left context (all previous tokens).
    """

    def __init__(self, model, config: Optional[StreamingConfig] = None):
        """
        Initialize the streaming generator.

        Args:
            model: Qwen3TTSModel instance
            config: Streaming configuration (optional)
        """
        self.model = model
        self.config = config or StreamingConfig()

        # Get decoder reference
        self.decoder = model.model.speech_tokenizer.model.decoder
        self.upsample_rate = self.decoder.total_upsample

        # High-priority CUDA stream for GPU scheduling priority over games
        self._cuda_stream = torch.cuda.Stream(priority=-1)

    def _prepare_generation(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
    ) -> dict:
        """Prepare inputs for generation."""
        # Build input text with assistant markers
        input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])

        # Prepare instruction if provided
        instruct_ids = [None]
        if instruct:
            instruct_ids = [self.model._tokenize_texts([self.model._build_instruct_text(instruct)])[0]]

        # Get generation kwargs with optional parameter overrides
        gen_kwargs = self.model._merge_generate_kwargs(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
        )

        return {
            'input_ids': input_ids,
            'instruct_ids': instruct_ids,
            'languages': [language],
            'speakers': [speaker],
            'non_streaming_mode': True,  # Still use full text mode
            **gen_kwargs,
        }

    def _run_generation(
        self,
        gen_kwargs: dict,
        token_buffer: ProgressiveTokenBuffer,
    ):
        """
        Run generation and capture tokens (runs in background thread).

        This is the key function that executes generation and captures
        codec tokens as they're produced via the codec_callback mechanism.
        """
        try:
            # Set high-priority CUDA stream on this thread for GPU scheduling
            torch.cuda.set_stream(self._cuda_stream)

            def on_codec_token(codec_ids: torch.Tensor):
                """Called by CodecStreamer after each generation step."""
                # codec_ids shape: (1, num_codebooks) - single token with all codebooks
                # Add to buffer for incremental decoding
                token_buffer.add_token(codec_ids.squeeze(0))

            logger.debug("[Generation] Starting model.generate() with codec_callback")

            # Run generation with streaming callback
            self.model.model.generate(
                codec_callback=on_codec_token,
                **gen_kwargs,
            )

            logger.debug("[Generation] model.generate() completed, total tokens: %d", token_buffer.token_count)
            token_buffer.mark_complete()

        except Exception as e:
            logger.error("[Generation] Exception in background generation thread:\n%s", traceback.format_exc())
            token_buffer.mark_error(e)

    def _decode_progressive(
        self,
        codes: torch.Tensor,
        samples_already_output: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode ALL tokens and return only the NEW audio portion.

        This is the key to artifact-free streaming: by always decoding from
        the beginning, each token has full left context. The audio for tokens
        [0:N] is identical whether decoded alone or as part of [0:N+M].

        Args:
            codes: All tokens (1, num_codebooks, total_tokens)
            samples_already_output: How many samples were already output

        Returns:
            Tuple of (new_audio, total_samples):
            - new_audio: Audio samples for the NEW portion only
            - total_samples: Total samples in the full decode (for tracking)
        """
        codes = codes.to(self.decoder.device)

        with torch.cuda.stream(self._cuda_stream):
            with torch.no_grad():
                # Decode all tokens
                wav = self.decoder(codes)

                # Get total samples from this decode
                total_samples = wav.shape[-1]

                # Only return audio after what we've already output
                new_audio = wav[..., samples_already_output:]

                return new_audio.squeeze().float().cpu().numpy(), total_samples

    async def generate_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio with true streaming using progressive batch decoding.

        Yields audio packets as codec tokens are generated. Each decode includes
        ALL tokens generated so far, but only the NEW audio portion is output.
        This guarantees identical quality to non-streaming batch decoding.

        Args:
            text: Text to synthesize
            speaker: Speaker name
            language: Language
            instruct: Optional instruction text
            temperature: Sampling temperature (higher = more expressive)
            top_k: Top-k sampling (lower = more focused)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repetitive patterns
            subtalker_temperature: Temperature for second-level codec generation
            subtalker_top_k: Top-k for second-level codec generation
            subtalker_top_p: Nucleus sampling for second-level codec generation

        Yields:
            PCM audio bytes (s16le, 24kHz, mono)
        """
        # Prepare generation inputs
        gen_kwargs = self._prepare_generation(
            text, speaker, language, instruct,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
        )

        # Create token buffer
        token_buffer = ProgressiveTokenBuffer(packet_size=self.config.packet_size)

        # Calculate required pre-buffer based on text length
        if self.config.pre_buffer_ms > 0:
            required_buffer_ms = self.config.pre_buffer_ms
        else:
            required_buffer_ms = estimate_required_buffer_ms(text)
        pre_buffer_samples = int(required_buffer_ms * self.config.sample_rate / 1000)

        # Start generation in background thread
        gen_thread = threading.Thread(
            target=self._run_generation,
            args=(gen_kwargs, token_buffer),
            daemon=True,
        )
        gen_thread.start()

        # Monitor buffer and yield packets
        start_time = time.perf_counter()
        packets_yielded = 0

        # Pre-buffer state
        pre_buffer: List[np.ndarray] = []
        buffered_samples = 0
        buffering_complete = False

        try:
            while True:
                # Check for errors - flush pre-buffer before raising
                if token_buffer.error:
                    logger.error("[Streaming] Generation error detected, flushing %d pre-buffered chunks before raising", len(pre_buffer))
                    if pre_buffer:
                        for chunk in pre_buffer:
                            if chunk.size > 0:
                                packets_yielded += 1
                                yield audio_to_pcm_bytes(chunk)
                        pre_buffer = []
                    raise token_buffer.error

                # Check if we have enough new tokens to decode
                should_decode = token_buffer.has_new_packet()

                # Also decode on completion if there are any remaining tokens
                if not should_decode and token_buffer.is_complete:
                    codes, decoded_token_count, _ = token_buffer.get_all_tokens()
                    if codes is not None and codes.shape[-1] > decoded_token_count:
                        should_decode = True

                if should_decode:
                    codes, decoded_token_count, output_sample_count = token_buffer.get_all_tokens()

                    if codes is not None:
                        total_tokens = codes.shape[-1]

                        if total_tokens > decoded_token_count:
                            # Decode ALL tokens, get only NEW audio (after samples already output)
                            audio, total_samples = self._decode_progressive(codes, output_sample_count)
                            token_buffer.mark_decoded(total_tokens, total_samples)

                            if audio.size > 0:
                                if not buffering_complete:
                                    # Accumulate in pre-buffer
                                    pre_buffer.append(audio)
                                    buffered_samples += len(audio)

                                    if buffered_samples >= pre_buffer_samples:
                                        # Flush pre-buffer
                                        buffer_time = (time.perf_counter() - start_time) * 1000
                                        logger.info("[Streaming] Pre-buffer ready: %d samples (%.0fms)", buffered_samples, buffer_time)
                                        for chunk in pre_buffer:
                                            packets_yielded += 1
                                            yield audio_to_pcm_bytes(chunk)
                                        pre_buffer = []
                                        buffering_complete = True
                                else:
                                    packets_yielded += 1
                                    yield audio_to_pcm_bytes(audio)

                if token_buffer.is_complete:
                    # Flush any remaining pre-buffer
                    if pre_buffer:
                        for chunk in pre_buffer:
                            if chunk.size > 0:
                                packets_yielded += 1
                                yield audio_to_pcm_bytes(chunk)
                    break
                else:
                    # Wait for more tokens
                    await asyncio.sleep(self.config.poll_interval)

        finally:
            # Ensure thread completes
            gen_thread.join(timeout=5.0)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("[Streaming] Generated %d packets in %.0fms", packets_yielded, elapsed)

    async def generate_streaming_with_stats(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[bytes, Optional[StreamingStats]], None]:
        """
        Generate with streaming and return stats with final packet.

        Same as generate_streaming but yields (bytes, stats) tuples.
        Stats are None for all packets except the last.
        Uses progressive batch decoding for artifact-free audio.
        """
        gen_kwargs = self._prepare_generation(
            text, speaker, language, instruct,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        token_buffer = ProgressiveTokenBuffer(packet_size=self.config.packet_size)

        # Calculate required pre-buffer based on text length
        if self.config.pre_buffer_ms > 0:
            required_buffer_ms = self.config.pre_buffer_ms
        else:
            required_buffer_ms = estimate_required_buffer_ms(text)
        pre_buffer_samples = int(required_buffer_ms * self.config.sample_rate / 1000)

        gen_thread = threading.Thread(
            target=self._run_generation,
            args=(gen_kwargs, token_buffer),
            daemon=True,
        )
        gen_thread.start()

        start_time = time.perf_counter()
        first_packet_time = None
        packets_yielded = 0
        total_tokens = 0
        total_decode_time = 0.0

        # Pre-buffer state
        pre_buffer: List[np.ndarray] = []
        buffered_samples = 0
        buffering_complete = False

        # Collect all chunks, yield all but last without stats
        all_chunks: List[bytes] = []

        try:
            while True:
                if token_buffer.error:
                    logger.error("[Streaming/Stats] Generation error detected, flushing pre-buffer before raising")
                    if pre_buffer:
                        for chunk in pre_buffer:
                            if chunk.size > 0:
                                packets_yielded += 1
                                yield audio_to_pcm_bytes(chunk), None
                        pre_buffer = []
                    if all_chunks:
                        for chunk_bytes in all_chunks:
                            packets_yielded += 1
                            yield chunk_bytes, None
                        all_chunks = []
                    raise token_buffer.error

                # Check if we have enough new tokens to decode
                should_decode = token_buffer.has_new_packet()

                # Also decode on completion if there are any remaining tokens
                if not should_decode and token_buffer.is_complete:
                    codes, decoded_token_count, _ = token_buffer.get_all_tokens()
                    if codes is not None and codes.shape[-1] > decoded_token_count:
                        should_decode = True

                if should_decode:
                    codes, decoded_token_count, output_sample_count = token_buffer.get_all_tokens()

                    if codes is not None:
                        current_total = codes.shape[-1]

                        if current_total > decoded_token_count:
                            decode_start = time.perf_counter()
                            audio, total_samples = self._decode_progressive(codes, output_sample_count)
                            decode_time = time.perf_counter() - decode_start
                            total_decode_time += decode_time

                            token_buffer.mark_decoded(current_total, total_samples)
                            total_tokens = current_total

                            if audio.size > 0:
                                audio_bytes = audio_to_pcm_bytes(audio)

                                if not buffering_complete:
                                    # Accumulate in pre-buffer
                                    pre_buffer.append(audio)
                                    buffered_samples += len(audio)

                                    if buffered_samples >= pre_buffer_samples:
                                        # Flush pre-buffer and record first packet time
                                        if first_packet_time is None:
                                            first_packet_time = time.perf_counter() - start_time

                                        # Yield all buffered chunks except last
                                        for i, chunk in enumerate(pre_buffer[:-1]):
                                            packets_yielded += 1
                                            yield audio_to_pcm_bytes(chunk), None

                                        # Keep last chunk for potential stats
                                        if pre_buffer:
                                            all_chunks.append(audio_to_pcm_bytes(pre_buffer[-1]))

                                        pre_buffer = []
                                        buffering_complete = True
                                else:
                                    # Yield previous chunk (if any), keep current for stats
                                    if all_chunks:
                                        packets_yielded += 1
                                        yield all_chunks[-1], None
                                        all_chunks = []

                                    if first_packet_time is None:
                                        first_packet_time = time.perf_counter() - start_time

                                    all_chunks.append(audio_bytes)

                if token_buffer.is_complete:
                    # Yield final chunk(s) with stats
                    if first_packet_time is None:
                        first_packet_time = time.perf_counter() - start_time

                    # Handle remaining pre-buffer (if buffering never completed)
                    if pre_buffer:
                        for i, chunk in enumerate(pre_buffer[:-1]):
                            packets_yielded += 1
                            yield audio_to_pcm_bytes(chunk), None
                        if pre_buffer:
                            all_chunks.append(audio_to_pcm_bytes(pre_buffer[-1]))

                    # Yield final chunk with stats
                    if all_chunks:
                        # Yield all but last without stats
                        for chunk_bytes in all_chunks[:-1]:
                            packets_yielded += 1
                            yield chunk_bytes, None

                        # Yield last with stats
                        packets_yielded += 1
                        stats = StreamingStats(
                            first_packet_time_ms=(first_packet_time or 0) * 1000,
                            total_packets=packets_yielded,
                            total_tokens=total_tokens,
                            total_time_ms=(time.perf_counter() - start_time) * 1000,
                            decode_time_ms=total_decode_time * 1000,
                        )
                        yield all_chunks[-1], stats

                    break
                else:
                    await asyncio.sleep(self.config.poll_interval)

        finally:
            gen_thread.join(timeout=5.0)


# Convenience function for simple streaming
async def stream_tts(
    model,
    text: str,
    speaker: str = "Serena",
    language: str = "English",
    instruct: Optional[str] = None,
    packet_size: int = 8,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Simple streaming TTS function.

    Args:
        model: Qwen3TTSModel instance
        text: Text to synthesize
        speaker: Speaker name
        language: Language
        instruct: Optional instruction
        packet_size: Tokens per audio packet
        temperature: Sampling temperature (higher = more expressive)
        top_k: Top-k sampling (lower = more focused)
        repetition_penalty: Penalty for repetitive patterns

    Yields:
        PCM audio bytes
    """
    config = StreamingConfig(packet_size=packet_size)
    generator = StreamingTTSGenerator(model, config)

    async for audio_bytes in generator.generate_streaming(
        text, speaker, language, instruct,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        yield audio_bytes
