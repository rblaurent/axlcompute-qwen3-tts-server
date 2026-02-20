"""
Qwen3-TTS Server - FastAPI wrapper for TTS synthesis.

Provides an HTTP API with emotion support using the qwen-tts package.

Endpoints:
- /synthesize: Standard synthesis (returns complete audio)
- /synthesize/stream: True streaming synthesis (~300ms first packet latency)
- /synthesize/wav: Returns complete WAV file
"""

import asyncio
import gc
import io
import logging
import os
import platform
import struct
import time
import re
import wave
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np
import torch
torch.set_float32_matmul_precision("high")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from emotion_mapper import map_emotion_to_instruct
from streaming_generator import StreamingTTSGenerator, StreamingConfig

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("qwen3_tts")
logger.setLevel(logging.DEBUG)

# Rotating file handler: 10MB max, 5 backups
_file_handler = RotatingFileHandler(
    "logs/server.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(_file_handler)

# Console handler (for when terminal is visible)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(_console_handler)

# Global model instance
model = None
model_name: str = ""
cuda_stream = None  # High-priority CUDA stream for GPU scheduling priority

# Reload state management
reload_lock = asyncio.Lock()
is_reloading = False


class SynthesizeRequest(BaseModel):
    """Request body for /synthesize endpoint."""
    text: str
    instruct: Optional[str] = None  # AI-generated voice instruction (preferred)
    emotion: str = "neutral"  # Fallback for emotion mapping
    intensity: float = 0.5  # Fallback for emotion mapping
    speaker: str = "Serena"
    language: str = "English"
    # Generation parameters (optional - use model defaults if not provided)
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    # Subtalker (second-level codec) generation parameters
    subtalker_temperature: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    # Adaptive pre-buffering: when true, wrap audio chunks in frames with RTF metadata
    adaptive_buffer: bool = False


class ReloadRequest(BaseModel):
    """Request body for /model/reload endpoint."""
    checkpoint_path: str


def get_model_from_env() -> str:
    """Get model path from environment variable or default."""
    env = os.environ.get("QWEN3_TTS_MODEL")
    if env:
        return env
    # Auto-detect: prefer WSL2 native copy if available
    if platform.system() == "Linux":
        wsl_path = os.path.expanduser("~/models/thalya-checkpoint")
        if os.path.exists(wsl_path):
            return wsl_path
    return "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads model on startup."""
    global model, model_name, cuda_stream

    model_name = get_model_from_env()
    logger.info("Loading Qwen3-TTS model: %s", model_name)

    try:
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("Model loaded successfully (flash_attention_2): %s", model_name)

        # Apply fork optimizations (Linux/WSL2 only)
        _apply_optimizations(model)

        # Create high-priority CUDA stream for GPU scheduling priority over games
        cuda_stream = torch.cuda.Stream(priority=-1)  # -1 = highest priority
        logger.info("Created high-priority CUDA stream (priority=-1)")

        logger.info(
            "GPU optimizations active: TF32=%s, flash_attention_2, streaming optimizations",
            torch.get_float32_matmul_precision(),
        )

        # Warmup to trigger torch.compile compilation (avoids cold-start latency)
        _warmup_model(model)

        # Log available speakers
        speakers = model.get_supported_speakers()
        languages = model.get_supported_languages()
        logger.info("Available speakers: %s", speakers)
        logger.info("Available languages: %s", languages)

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        model = None

    yield

    # Cleanup on shutdown
    if model is not None:
        del model
        torch.cuda.empty_cache()


def _apply_optimizations(model):
    """Apply dffdeeq fork optimizations: compiled decoder, fast codebook, compiled CodePredictor.

    Only works on Linux with Triton installed.
    """
    if platform.system() == "Windows":
        logger.info("Skipping optimizations (Windows — Triton not available)")
        return

    if os.environ.get("NO_COMPILE"):
        logger.info("Skipping optimizations (NO_COMPILE set)")
        return

    try:
        model.enable_streaming_optimizations(
            decode_window_frames=80,
            use_compile=True,
            use_cuda_graphs=True,
            compile_mode="reduce-overhead",
            use_fast_codebook=True,
            compile_codebook_predictor=True,
        )
        logger.info("Streaming optimizations applied (fast_codebook + compiled decoder + compiled CodePredictor)")
    except Exception as e:
        logger.warning("Optimizations failed, falling back to unoptimized: %s", e)


def _warmup_model(model):
    """Run warmup generations to trigger torch.compile compilation."""
    logger.info("Running warmup generation (triggers torch.compile — this may take 30-120s)...")
    try:
        speaker = model.get_supported_speakers()[0]
        # Batch warmup
        t0 = time.monotonic()
        model.generate_custom_voice(
            text="Warmup.",
            language="English",
            speaker=speaker,
        )
        t1 = time.monotonic()
        logger.info("Batch warmup done in %.1fs (includes torch.compile)", t1 - t0)
        # Streaming warmup (consume the generator to trigger compile on streaming path)
        if platform.system() != "Windows" and hasattr(model, "stream_generate_custom_voice"):
            t2 = time.monotonic()
            for _ in model.stream_generate_custom_voice(
                text="Warmup.",
                language="English",
                speaker=speaker,
            ):
                pass
            t3 = time.monotonic()
            logger.info("Streaming warmup done in %.1fs", t3 - t2)
        logger.info("Warmup complete — total %.1fs", time.monotonic() - t0)
    except Exception as e:
        logger.warning("Warmup failed (non-fatal): %s", e)


app = FastAPI(
    title="Qwen3-TTS Server",
    description="TTS API using qwen-tts package with emotion support",
    version="1.0.0",
    lifespan=lifespan
)


def strip_emotion_tags(text: str) -> str:
    """Remove emotion tags like [excited], [sarcastic] from text."""
    return re.sub(r'\[[^\]]+\]\s*', '', text)


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert audio numpy array to WAV bytes."""
    # Ensure float32
    audio = audio_array.astype(np.float32)

    # Normalize to [-1, 1] range if needed
    max_val = max(abs(audio.max()), abs(audio.min()))
    if max_val > 1.0:
        audio = audio / max_val

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def audio_to_pcm_bytes(audio_array: np.ndarray) -> bytes:
    """Convert audio numpy array to raw PCM bytes (16-bit mono)."""
    if audio_array.size == 0:
        return b""

    # Ensure float32
    audio = audio_array.astype(np.float32)

    # Normalize to [-1, 1] range if needed
    max_val = max(abs(audio.max()), abs(audio.min()))
    if max_val > 1.0:
        audio = audio / max_val

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": model_name if model is not None else None,
        "streaming": True,  # True streaming via /synthesize/stream endpoint
        "backend": "qwen-tts"
    }


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text with emotion control.

    Returns raw PCM audio (s16le, 24kHz, mono).
    """
    if is_reloading:
        raise HTTPException(
            status_code=503,
            detail="Model is being reloaded. Please try again shortly."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    # Strip emotion tags from text
    clean_text = strip_emotion_tags(request.text)

    if not clean_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after stripping emotion tags."
        )

    # Use AI-generated instruction if provided, otherwise fall back to emotion mapping
    if request.instruct and request.instruct.strip():
        instruct = request.instruct.strip()
        logger.info("[Synthesize] Text: %s...", clean_text[:100])
        logger.info("[Synthesize] Using AI-generated instruct: %s", instruct)
    else:
        instruct = map_emotion_to_instruct(request.emotion, request.intensity)
        logger.info("[Synthesize] Text: %s...", clean_text[:100])
        logger.info("[Synthesize] Emotion: %s, Intensity: %s", request.emotion, request.intensity)
        logger.info("[Synthesize] Mapped instruct: %s", instruct or '(none)')

    logger.info("[Synthesize] Speaker: %s, Language: %s", request.speaker, request.language)

    try:
        # Generate audio using qwen-tts with high-priority CUDA stream
        with torch.cuda.stream(cuda_stream):
            wavs, sample_rate = model.generate_custom_voice(
                text=clean_text,
                language=request.language,
                speaker=request.speaker,
                instruct=instruct if instruct else None,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                subtalker_temperature=request.subtalker_temperature,
                subtalker_top_k=request.subtalker_top_k,
                subtalker_top_p=request.subtalker_top_p,
            )

        if not wavs or len(wavs) == 0:
            raise HTTPException(
                status_code=500,
                detail="No audio generated."
            )

        # Convert to PCM bytes
        audio_array = wavs[0]
        pcm_bytes = audio_to_pcm_bytes(audio_array)

        logger.info("[Synthesize] Generated %d bytes of audio", len(pcm_bytes))

        return StreamingResponse(
            io.BytesIO(pcm_bytes),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": str(sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[Synthesize] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/wav")
async def synthesize_wav(request: SynthesizeRequest):
    """
    Synthesize speech from text with emotion control.

    Returns complete WAV file.
    """
    if is_reloading:
        raise HTTPException(
            status_code=503,
            detail="Model is being reloaded. Please try again shortly."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    # Strip emotion tags from text
    clean_text = strip_emotion_tags(request.text)

    if not clean_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after stripping emotion tags."
        )

    # Use AI-generated instruction if provided, otherwise fall back to emotion mapping
    if request.instruct and request.instruct.strip():
        instruct = request.instruct.strip()
        logger.info("[Synthesize/WAV] Text: %s...", clean_text[:100])
        logger.info("[Synthesize/WAV] Using AI-generated instruct: %s", instruct)
    else:
        instruct = map_emotion_to_instruct(request.emotion, request.intensity)
        logger.info("[Synthesize/WAV] Text: %s...", clean_text[:100])
        logger.info("[Synthesize/WAV] Emotion: %s, Intensity: %s", request.emotion, request.intensity)
        logger.info("[Synthesize/WAV] Mapped instruct: %s", instruct or '(none)')

    try:
        # Generate audio using qwen-tts with high-priority CUDA stream
        with torch.cuda.stream(cuda_stream):
            wavs, sample_rate = model.generate_custom_voice(
                text=clean_text,
                language=request.language,
                speaker=request.speaker,
                instruct=instruct if instruct else None,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                subtalker_temperature=request.subtalker_temperature,
                subtalker_top_k=request.subtalker_top_k,
                subtalker_top_p=request.subtalker_top_p,
            )

        if not wavs or len(wavs) == 0:
            raise HTTPException(
                status_code=500,
                detail="No audio generated."
            )

        # Convert to WAV bytes
        audio_array = wavs[0]
        wav_bytes = audio_to_wav_bytes(audio_array, sample_rate)

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[Synthesize/WAV] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Synthesize speech with true token-level streaming.

    Returns chunked PCM audio as packets become available.
    First packet latency is approximately 300-400ms.

    Audio format: s16le, 24kHz, mono
    Each packet contains approximately 300ms of audio.
    """
    if is_reloading:
        raise HTTPException(
            status_code=503,
            detail="Model is being reloaded. Please try again shortly."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    # Strip emotion tags from text
    clean_text = strip_emotion_tags(request.text)

    if not clean_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after stripping emotion tags."
        )

    # Use AI-generated instruction if provided, otherwise fall back to emotion mapping
    if request.instruct and request.instruct.strip():
        instruct = request.instruct.strip()
        logger.info("[Stream] Text: %s...", clean_text[:100])
        logger.info("[Stream] Using AI-generated instruct: %s", instruct)
    else:
        instruct = map_emotion_to_instruct(request.emotion, request.intensity)
        logger.info("[Stream] Text: %s...", clean_text[:100])
        logger.info("[Stream] Emotion: %s, Intensity: %s", request.emotion, request.intensity)
        logger.info("[Stream] Mapped instruct: %s", instruct or '(none)')

    logger.info("[Stream] Speaker: %s, Language: %s", request.speaker, request.language)

    use_fork_streaming = (
        platform.system() != "Windows"
        and hasattr(model, "stream_generate_custom_voice")
    )

    if use_fork_streaming:
        async def audio_stream():
            """Async generator using fork's optimized streaming."""
            loop = asyncio.get_event_loop()
            gen = model.stream_generate_custom_voice(
                text=clean_text,
                speaker=request.speaker,
                language=request.language,
                instruct=instruct if instruct else None,
                emit_every_frames=12,
                decode_window_frames=80,
                use_optimized_decode=False,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                subtalker_temperature=request.subtalker_temperature,
                subtalker_top_k=request.subtalker_top_k,
                subtalker_top_p=request.subtalker_top_p,
            )
            try:
                while True:
                    result = await loop.run_in_executor(None, next, gen, None)
                    if result is None:
                        break
                    audio_chunk, sr = result
                    yield audio_to_pcm_bytes(audio_chunk)
            except StopIteration:
                pass
            except Exception as e:
                logger.exception("[Stream] Error during streaming: %s", e)
                raise
    else:
        # Windows fallback: use old StreamingTTSGenerator with codec_callback
        config = StreamingConfig(
            packet_size=4,
        )
        generator = StreamingTTSGenerator(model, config)

        async def audio_stream():
            """Async generator using old codec_callback streaming."""
            try:
                async for audio_bytes in generator.generate_streaming(
                    text=clean_text,
                    speaker=request.speaker,
                    language=request.language,
                    instruct=instruct if instruct else None,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    subtalker_temperature=request.subtalker_temperature,
                    subtalker_top_k=request.subtalker_top_k,
                    subtalker_top_p=request.subtalker_top_p,
                ):
                    yield audio_bytes
            except Exception as e:
                logger.exception("[Stream] Error during streaming: %s", e)
                raise

    # Wrap in framed stream with RTF metadata when adaptive_buffer is requested
    response_headers = {
        "X-Audio-Sample-Rate": "24000",
        "X-Audio-Channels": "1",
        "X-Audio-Format": "s16le",
        "X-Streaming": "true",
        "Cache-Control": "no-cache",
        "Transfer-Encoding": "chunked",
    }

    if request.adaptive_buffer:
        raw_gen = audio_stream()

        async def framed_stream():
            start_time = time.monotonic()
            total_audio_samples = 0
            async for pcm_bytes in raw_gen:
                total_audio_samples += len(pcm_bytes) // 2  # 16-bit samples
                audio_duration = total_audio_samples / 24000.0
                elapsed = time.monotonic() - start_time
                rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
                # Frame: [uint32 pcm_data_length][float32 cumulative_rtf][pcm_data]
                header = struct.pack('<If', len(pcm_bytes), rtf)
                yield header + pcm_bytes

        response_gen = framed_stream()
        response_headers["X-Framed"] = "true"
        logger.info("[Stream] Adaptive buffer enabled, using framed streaming")
    else:
        response_gen = audio_stream()

    return StreamingResponse(
        response_gen,
        media_type="audio/pcm",
        headers=response_headers
    )


@app.get("/speakers")
async def list_speakers():
    """List available speakers for the loaded model."""
    if model is None:
        # Return defaults if model not loaded
        speakers = [
            {"name": "Serena", "description": "Warm, gentle young female (Chinese native)", "language": "zh/en"},
            {"name": "Aiden", "description": "Sunny American male", "language": "en"},
            {"name": "Ryan", "description": "Dynamic male with rhythmic drive", "language": "en"},
            {"name": "Vivian", "description": "Bright, edgy young female (Chinese native)", "language": "zh/en"},
        ]
    else:
        # Get from model
        speaker_names = model.get_supported_speakers()
        speakers = [{"name": name, "description": "", "language": ""} for name in speaker_names]

    return {
        "speakers": speakers,
        "default": "Serena"
    }


@app.get("/languages")
async def list_languages():
    """List available languages for the loaded model."""
    if model is None:
        languages = ["English", "Chinese", "French", "Japanese", "Korean"]
    else:
        languages = model.get_supported_languages()

    return {
        "languages": languages,
        "default": "English"
    }


@app.post("/model/reload")
async def reload_model(request: ReloadRequest):
    """
    Reload the TTS model with a different checkpoint.

    This allows switching between fine-tuned voice models at runtime.
    The endpoint acquires a lock to prevent concurrent reloads and
    properly releases CUDA memory before loading the new model.
    """
    global model, model_name, is_reloading

    checkpoint_path = request.checkpoint_path.strip()
    if not checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail="checkpoint_path is required"
        )

    # Determine if this is a HuggingFace model ID (e.g., "rblaurent/qwen3-tts-springs")
    # vs a local filesystem path. HF IDs contain "/" but no OS path separators like "\" or start with "/".
    is_hf_model = "/" in checkpoint_path and not os.path.isabs(checkpoint_path) and "\\" not in checkpoint_path

    # Check if local path exists (skip for HF model IDs — they'll be downloaded by from_pretrained)
    if not is_hf_model and not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint path does not exist: {checkpoint_path}"
        )

    # Acquire lock to prevent concurrent reloads
    async with reload_lock:
        is_reloading = True
        logger.info("[Reload] Starting model reload from: %s", checkpoint_path)

        try:
            # Delete current model and free CUDA memory
            if model is not None:
                logger.info("[Reload] Unloading current model...")
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("[Reload] CUDA memory cleared")

            # Load new model
            logger.info("[Reload] Loading new model from: %s", checkpoint_path)
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained(
                checkpoint_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            model_name = checkpoint_path

            # Apply fork optimizations (Linux/WSL2 only)
            _apply_optimizations(model)

            # Get available speakers
            speakers = model.get_supported_speakers()
            languages = model.get_supported_languages()

            logger.info("[Reload] Model loaded successfully: %s", checkpoint_path)
            logger.info("[Reload] Available speakers: %s", speakers)
            logger.info("[Reload] Available languages: %s", languages)

            return {
                "status": "ok",
                "model_path": checkpoint_path,
                "speakers": speakers,
                "languages": languages
            }

        except Exception as e:
            logger.exception("[Reload] Error loading model: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )

        finally:
            is_reloading = False


@app.get("/model/info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        return {
            "status": "not_loaded",
            "model_path": None,
            "speakers": [],
            "languages": [],
            "is_reloading": is_reloading
        }

    return {
        "status": "loaded",
        "model_path": model_name,
        "speakers": model.get_supported_speakers(),
        "languages": model.get_supported_languages(),
        "is_reloading": is_reloading
    }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host=host, port=port)
