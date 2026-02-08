# GPU Optimization Findings for Qwen3-TTS Server

**Date:** 2026-02-08
**Hardware:** RTX 5090, Windows 11, Python 3.11, PyTorch 2.7.0+cu128
**Baseline:** RTF ~5x streaming, ~2x batch

---

## 1. What We Tried

### A. Server-side optimizations (applied to `server.py`)

| Change | What it does | Result |
|--------|-------------|--------|
| `torch.set_float32_matmul_precision("high")` | Enables TF32 on Ampere+ tensor cores | Negligible impact |
| `attn_implementation="flash_attention_2"` | Flash Attention for LLM layers | Negligible impact on RTF |
| Manual CUDA graph capture for decoder | Pre-records decoder GPU ops for lengths 1..60, replays instead of launching kernels | Fixed streaming crash (InferenceMode error), marginal RTF improvement |

**Results after all 3 optimizations:**
- Streaming RTF: ~1.6-1.98x (was ~5x)
- Batch RTF: ~1.83x (was ~2x)
- Streaming TTFB: 6-9 seconds (unchanged, very bad)

**Note:** The streaming RTF improvement from ~5x to ~1.8x may have been from other recent changes, not these optimizations specifically. The batch RTF barely moved (~2x to ~1.83x).

### B. Reference fork (rekuenkdr/Qwen3-TTS-streaming)

Tested the community fork that adds streaming directly into the model's generation loop.

**Architecture differences:**
- Streaming is built into `stream_generate_pcm()` inside the model itself
- Yields audio chunks *during* the autoregressive LLM token generation
- Uses `torch.compile(mode="reduce-overhead")` for the decoder (includes CUDA graphs automatically)
- Sliding window decode with overlap/crossfade between chunks

**Results (without torch.compile, since Triton is broken on Windows):**
- Streaming RTF: ~1.75-2.1x (same as our server)
- Batch RTF: ~1.68-1.69x (same as our server)
- Streaming TTFB: **526ms-1.5s** (vs our 6-9s)

**Audio quality issues with the fork:**
- Streaming audio sounds awful (artifacts, distortion)
- Batch audio defaulted to wrong language (Russian instead of French)
- The fork's `stream_generate_pcm` may not handle CustomVoice fine-tuned models correctly
- Only `stream_generate_voice_clone` (Base model) was properly implemented; CustomVoice streaming was not in the fork

---

## 2. Root Cause Analysis

### Why RTF is ~1.8x (not 0.5x)

The bottleneck is the **autoregressive LLM token generation** (the "talker" model), NOT the decoder:
- The decoder runs in ~10% of total time
- The LLM generates tokens one at a time in a loop using HuggingFace `generate()`
- Flash attention and TF32 help the LLM marginally, but don't change the fundamental O(n) autoregressive loop
- **`torch.compile` with `reduce-overhead` mode** is the known fix — it fuses operations and uses CUDA graphs for the LLM itself, achieving ~2-3x speedup on the token generation

### Why torch.compile doesn't work on Windows

PyTorch 2.7.0+cu128 requires Triton for the `inductor` backend. On Windows:
- `triton-windows` 3.5.1: Missing `triton_key` import → `ImportError`
- `triton-windows` 3.4.0: Missing `launch_enter_hook` → `AttributeError`
- No other Triton versions available for Windows
- This is a known PyTorch/Windows/Triton compatibility issue with no current fix

### Why our TTFB is 6-9 seconds

Our streaming architecture in `streaming_generator.py`:
1. Starts the full `model.generate()` call
2. Uses a `codec_callback` to intercept tokens as they're produced
3. Accumulates tokens in a buffer, decodes when `packet_size` tokens collected
4. BUT: The LLM must generate enough tokens before the first callback fires

The fork's architecture:
1. Runs the LLM token generation loop directly (not via HF `generate()`)
2. Yields decoded audio every `emit_every_frames` tokens
3. First chunk arrives after just 4-8 tokens (~500ms)

Our approach has higher TTFB because HF `generate()` has overhead before the first token callback fires.

---

## 3. Recommendations

### Priority 1: Fix torch.compile (highest impact on RTF)

**Option A: Use WSL2/Linux** (recommended, easiest)
- Triton works natively on Linux
- Run the TTS server inside WSL2 with GPU passthrough
- Expected: RTF drops from ~1.8x to ~0.5-0.7x

**Option B: Fix Triton on Windows**
- Wait for a compatible `triton-windows` release
- Or build Triton from source matching PyTorch 2.7
- High effort, uncertain timeline

**Option C: Use a different compilation backend**
- Try `torch.compile(backend="eager")` — no Triton needed, but less optimization
- Try ONNX Runtime or TensorRT for the decoder
- Moderate effort, moderate gain

### Priority 2: Improve TTFB (highest impact on perceived latency)

**Option A: Port the fork's streaming architecture into our server**
- Replace our `codec_callback` approach with direct token loop + incremental decode
- Challenge: need to handle CustomVoice models correctly (the fork only did Base model streaming)
- This would get TTFB from 6-9s down to ~500ms-1.5s

**Option B: Use vLLM/SGLang for LLM inference**
- These frameworks are optimized for autoregressive generation
- Would replace HuggingFace `generate()` with a much faster inference engine
- Highest potential speedup (RTF + TTFB), but significant integration work

### Priority 3: Decoder optimization (minor impact)

The decoder is already fast relative to the LLM. Further optimizations:
- `torch.compile` the decoder separately (once Triton works)
- ONNX/TensorRT export for the decoder
- These would help streaming TTFB slightly but won't change overall RTF much

---

## 4. Summary Table

| Approach | RTF Impact | TTFB Impact | Effort | Status |
|----------|-----------|-------------|--------|--------|
| TF32 + Flash Attention | Negligible | None | Done | Applied |
| Manual CUDA graphs (decoder) | Negligible | None | Done | Applied, but questionable value |
| `torch.compile` (needs Triton) | **~2-3x speedup** | Moderate | Blocked on Windows | Need Linux/WSL2 |
| Port fork's streaming loop | None | **~10x improvement** | Medium | Not started |
| vLLM/SGLang backend | **~2-4x speedup** | **~5-10x improvement** | High | Not started |

**Bottom line:** To reach RTF <0.5, we need `torch.compile` working (easiest via WSL2/Linux). To fix TTFB, we need to rearchitect the streaming loop. Both are independent and can be done in parallel.
