"""Patch the installed fork to add stream_generate_custom_voice() method."""
import sys
import importlib.util

# Auto-detect the installed package location
spec = importlib.util.find_spec("qwen_tts.inference.qwen3_tts_model")
if spec is None or spec.origin is None:
    print("ERROR: Could not find qwen_tts.inference.qwen3_tts_model package")
    sys.exit(1)

FILE = spec.origin
print(f"Found model file at: {FILE}")

with open(FILE, "r") as f:
    content = f.read()

MARKER = "    # custom voice model\n    @torch.no_grad()\n    def generate_custom_voice("

if "stream_generate_custom_voice" in content:
    print("stream_generate_custom_voice already exists, skipping")
    sys.exit(0)

if MARKER not in content:
    print(f"ERROR: Could not find insertion marker in {FILE}")
    sys.exit(1)

NEW_METHOD = '''\
    # custom voice model - streaming
    def stream_generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = None,
        instruct: Optional[str] = None,
        non_streaming_mode: bool = False,
        # Streaming control
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 0,
        max_frames: int = 10000,
        # Optimization
        use_optimized_decode: bool = True,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Stream custom voice speech generation, yielding PCM chunks as they are generated.

        NOTE: This method only supports single-sample generation (no batching).

        Args:
            text: Text to synthesize (single string only).
            speaker: Speaker name.
            language: Language for synthesis.
            instruct: Optional instruction text for voice control.
            non_streaming_mode: Whether to use non-streaming text input mode.
            emit_every_frames: Emit PCM chunk every N codec frames.
            decode_window_frames: Window size for decoding.
            overlap_samples: Overlap samples for crossfade between chunks.
            max_frames: Maximum codec frames to generate.
            use_optimized_decode: Use CUDA graph optimized decode when available.
            **kwargs: Generation parameters (do_sample, top_k, top_p, temperature, etc.)

        Yields:
            Tuple[np.ndarray, int]: (pcm_chunk as float32 array, sample_rate)
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support stream_generate_custom_voice"
            )

        if isinstance(text, list):
            raise ValueError("stream_generate_custom_voice only supports single text, not batch")

        texts = [text]
        languages = [language if language is not None else "Auto"]
        speakers = [speaker]
        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(texts[0])])

        instruct_ids: List[Optional[torch.Tensor]] = []
        if instruct is not None and instruct != "":
            instruct_ids.append(self._tokenize_texts([self._build_instruct_text(instruct)])[0])
        else:
            instruct_ids.append(None)

        # Merge kwargs with defaults, then filter to stream_generate_pcm supported params
        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        supported_params = {
            "do_sample", "top_k", "top_p", "temperature",
            "repetition_penalty",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_params}

        for chunk, sr in self.model.stream_generate_pcm(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
            use_optimized_decode=use_optimized_decode,
            **gen_kwargs,
        ):
            yield chunk, sr

    # custom voice model - batch
    @torch.no_grad()
    def generate_custom_voice('''

content = content.replace(MARKER, NEW_METHOD)

with open(FILE, "w") as f:
    f.write(content)

print("Patched successfully")
