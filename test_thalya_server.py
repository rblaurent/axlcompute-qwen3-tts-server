#!/usr/bin/env python3
"""
Test script for Qwen3-TTS server with the Thalya voice (French, fine-tuned).

Supports three test suites:
  - styles:  Different style instructions on representative French texts
  - params:  Parameter sweeps (temperature, top_k, repetition_penalty, etc.)
  - timing:  Streaming vs batch comparison on short/medium/long texts

Usage:
  python test_thalya_server.py                          # Run all suites
  python test_thalya_server.py --suite styles            # Style tests only
  python test_thalya_server.py --suite params            # Parameter sweep only
  python test_thalya_server.py --suite timing            # Timing benchmark
  python test_thalya_server.py --custom --text "Bonjour" # Single custom test
"""

import argparse
import datetime
import os
import sys
import time
import wave

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEAKER = "thalya"
LANGUAGE = "French"
THALYA_CHECKPOINT = "T:/Projects/Qwen3-TTS/thalya/model/checkpoint"
SAMPLE_RATE = 24000
OUTPUT_ROOT = "test_results"

DEFAULT_URL = "http://127.0.0.1:8765"

# Default generation parameters (server defaults when None)
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dir(output_dir: str | None = None) -> str:
    """Create and return a timestamped output directory for this run."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_ROOT, f"thalya_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def save_wav(pcm_bytes: bytes, path: str, sample_rate: int = SAMPLE_RATE):
    """Save raw s16le PCM bytes as a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def audio_duration_s(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Compute audio duration from raw s16le PCM bytes."""
    num_samples = len(pcm_bytes) // 2  # 16-bit = 2 bytes per sample
    return num_samples / sample_rate


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
    """Print a simple aligned table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    print()


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

def check_server(base_url: str) -> bool:
    """Check if the server is reachable."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"Server status: {data.get('status', '?')}, model_loaded: {data.get('model_loaded')}")
        return True
    except Exception as e:
        print(f"Server not reachable at {base_url}: {e}")
        return False


def ensure_thalya_loaded(base_url: str):
    """Check if thalya model is loaded; if not, reload it."""
    try:
        r = requests.get(f"{base_url}/model/info", timeout=5)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        print(f"Failed to get model info: {e}")
        sys.exit(1)

    model_path = info.get("model_path", "") or ""
    # Check if thalya checkpoint is already loaded
    if "thalya" in model_path.lower() or "checkpoint" in model_path.lower():
        print(f"Thalya model already loaded: {model_path}")
        return

    print(f"Current model: {model_path or '(none)'}")
    print(f"Reloading thalya from: {THALYA_CHECKPOINT}")
    try:
        r = requests.post(
            f"{base_url}/model/reload",
            json={"checkpoint_path": THALYA_CHECKPOINT},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        print(f"Reload result: {data.get('status', '?')} - {data.get('model_path', '?')}")
    except Exception as e:
        print(f"Failed to reload model: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def build_payload(
    text: str,
    instruct: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    subtalker_temperature: float | None = None,
    subtalker_top_k: int | None = None,
    subtalker_top_p: float | None = None,
) -> dict:
    payload = {
        "text": text,
        "speaker": SPEAKER,
        "language": LANGUAGE,
    }
    if instruct is not None:
        payload["instruct"] = instruct
    if temperature is not None:
        payload["temperature"] = temperature
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if subtalker_temperature is not None:
        payload["subtalker_temperature"] = subtalker_temperature
    if subtalker_top_k is not None:
        payload["subtalker_top_k"] = subtalker_top_k
    if subtalker_top_p is not None:
        payload["subtalker_top_p"] = subtalker_top_p
    return payload


def synthesize_streaming(base_url: str, payload: dict) -> dict:
    """
    Call /synthesize/stream and collect PCM audio with timing metrics.
    Returns dict with keys: pcm_bytes, ttfb_ms, total_ms, audio_duration_s,
                            realtime_factor, packets
    """
    url = f"{base_url}/synthesize/stream"
    t_start = time.perf_counter()
    ttfb = None
    packets = 0
    chunks = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    if ttfb is None:
                        ttfb = (time.perf_counter() - t_start) * 1000
                    chunks.append(chunk)
                    packets += 1
    except Exception as e:
        return {"error": str(e)}

    t_total = (time.perf_counter() - t_start) * 1000
    pcm = b"".join(chunks)
    dur = audio_duration_s(pcm)

    return {
        "pcm_bytes": pcm,
        "ttfb_ms": ttfb or 0,
        "total_ms": t_total,
        "audio_duration_s": dur,
        "realtime_factor": (t_total / (dur * 1000)) if dur > 0 else float("inf"),
        "packets": packets,
    }


def synthesize_batch(base_url: str, payload: dict) -> dict:
    """
    Call /synthesize (batch) and collect PCM audio with timing metrics.
    Returns dict with keys: pcm_bytes, ttfb_ms, total_ms, audio_duration_s,
                            realtime_factor
    """
    url = f"{base_url}/synthesize"
    t_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        pcm = resp.content
    except Exception as e:
        return {"error": str(e)}

    t_total = (time.perf_counter() - t_start) * 1000
    dur = audio_duration_s(pcm)

    return {
        "pcm_bytes": pcm,
        "ttfb_ms": t_total,  # batch: TTFB == total time
        "total_ms": t_total,
        "audio_duration_s": dur,
        "realtime_factor": (t_total / (dur * 1000)) if dur > 0 else float("inf"),
        "packets": 1,
    }


def synthesize(base_url: str, payload: dict, stream: bool = True) -> dict:
    """Dispatch to streaming or batch synthesis."""
    if stream:
        return synthesize_streaming(base_url, payload)
    else:
        return synthesize_batch(base_url, payload)


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

STYLE_TESTS = [
    {
        "name": "neutral",
        "instruct": None,
        "text": "Bonjour, je suis ravie de vous accueillir aujourd'hui. Comment puis-je vous aider?",
    },
    {
        "name": "happy",
        "instruct": "Parle avec joie et enthousiasme, comme si tu venais de recevoir une bonne nouvelle.",
        "text": "Quelle merveilleuse surprise! Je suis tellement contente de te voir, c'est formidable!",
    },
    {
        "name": "sad",
        "instruct": "Parle avec tristesse et emotion, d'une voix douce et melancolique.",
        "text": "Je me souviens encore de ce jour-la. Tout semblait si fragile, si ephemere. Le temps passe trop vite.",
    },
    {
        "name": "angry",
        "instruct": "Parle avec colere et frustration, d'une voix forte et determinee.",
        "text": "C'est absolument inacceptable! Je refuse de laisser passer une chose pareille, il faut que ca change!",
    },
    {
        "name": "whisper",
        "instruct": "Parle en chuchotant, d'une voix tres douce et intime, presque secrete.",
        "text": "Ecoute bien ce que je vais te dire. C'est un secret, personne d'autre ne doit le savoir.",
    },
    {
        "name": "sarcastic",
        "instruct": "Parle avec sarcasme et ironie, d'un ton moqueur et detache.",
        "text": "Oh mais bien sur, quelle idee absolument geniale. Je suis impressionnee, vraiment, bravo.",
    },
    {
        "name": "confident",
        "instruct": "Parle avec confiance et autorite, d'une voix posee et affirmee.",
        "text": "Nous allons reussir ce projet. J'ai confiance en notre equipe et en notre strategie. Avancez sans hesiter.",
    },
    {
        "name": "playful",
        "instruct": "Parle de maniere espiegle et taquine, avec un sourire dans la voix.",
        "text": "Alors, tu pensais vraiment pouvoir me surprendre? Je te connais mieux que tu ne le crois, petit malin!",
    },
]


def run_styles_suite(base_url: str, output_dir: str, stream: bool = True):
    """Run style instruction tests."""
    mode = "streaming" if stream else "batch"
    print("\n" + "=" * 60)
    print(f"STYLE INSTRUCTION SUITE ({mode})")
    print("=" * 60)

    rows = []
    for test in STYLE_TESTS:
        name = test["name"]
        print(f"\n  [{name}] ", end="", flush=True)

        payload = build_payload(text=test["text"], instruct=test["instruct"])
        result = synthesize(base_url, payload, stream=stream)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            rows.append([name, "ERROR", "-", "-", "-"])
            continue

        wav_path = os.path.join(output_dir, f"styles_{name}.wav")
        save_wav(result["pcm_bytes"], wav_path)

        print(f"TTFB={result['ttfb_ms']:.0f}ms  total={result['total_ms']:.0f}ms  "
              f"audio={result['audio_duration_s']:.1f}s  RT={result['realtime_factor']:.2f}x  "
              f"packets={result['packets']}")

        rows.append([
            name,
            f"{result['ttfb_ms']:.0f}",
            f"{result['total_ms']:.0f}",
            f"{result['audio_duration_s']:.1f}",
            f"{result['realtime_factor']:.2f}",
        ])

    print("\n--- Style Suite Summary ---")
    print_table(
        ["Style", "TTFB(ms)", "Total(ms)", "Audio(s)", "RT Factor"],
        rows,
    )


PARAM_TEXT = "Bonjour, je suis Thalya. Aujourd'hui nous allons explorer ensemble les possibilites de la synthese vocale. C'est un domaine fascinant qui ne cesse de progresser."
PARAM_INSTRUCT = "Parle de maniere naturelle et engageante."

PARAM_SWEEPS = [
    ("temperature", "temperature", [0.3, 0.6, 0.9, 1.2]),
    ("top_k", "top_k", [10, 30, 50, 100]),
    ("rep_penalty", "repetition_penalty", [1.0, 1.05, 1.15]),
    ("sub_temp", "subtalker_temperature", [0.5, 0.9, 1.2]),
]


def run_params_suite(base_url: str, output_dir: str, stream: bool = True):
    """Run parameter sweep tests."""
    mode = "streaming" if stream else "batch"
    print("\n" + "=" * 60)
    print(f"PARAMETER SWEEP SUITE ({mode})")
    print("=" * 60)

    for sweep_label, param_name, values in PARAM_SWEEPS:
        print(f"\n--- Sweep: {param_name} ---")
        rows = []

        for val in values:
            tag = f"{sweep_label}_{val}"
            print(f"  [{tag}] ", end="", flush=True)

            kwargs = {param_name: val}
            payload = build_payload(
                text=PARAM_TEXT,
                instruct=PARAM_INSTRUCT,
                **kwargs,
            )
            result = synthesize(base_url, payload, stream=stream)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                rows.append([str(val), "ERROR", "-", "-", "-"])
                continue

            wav_path = os.path.join(output_dir, f"params_{tag}.wav")
            save_wav(result["pcm_bytes"], wav_path)

            print(f"TTFB={result['ttfb_ms']:.0f}ms  total={result['total_ms']:.0f}ms  "
                  f"audio={result['audio_duration_s']:.1f}s  RT={result['realtime_factor']:.2f}x")

            rows.append([
                str(val),
                f"{result['ttfb_ms']:.0f}",
                f"{result['total_ms']:.0f}",
                f"{result['audio_duration_s']:.1f}",
                f"{result['realtime_factor']:.2f}",
            ])

        print_table(
            [param_name, "TTFB(ms)", "Total(ms)", "Audio(s)", "RT Factor"],
            rows,
        )


TIMING_TEXTS = [
    (
        "10w",
        "Bonjour, comment allez-vous aujourd'hui? J'espere que tout va bien.",
    ),
    (
        "20w",
        "La synthese vocale est une technologie fascinante qui permet de convertir "
        "du texte ecrit en parole naturelle et expressive pour tous.",
    ),
    (
        "30w",
        "Aujourd'hui nous allons explorer ensemble les possibilites offertes par "
        "l'intelligence artificielle dans le domaine de la synthese vocale. "
        "C'est un sujet passionnant qui ne cesse d'evoluer et de surprendre.",
    ),
    (
        "40w",
        "L'intelligence artificielle a fait des progres considerables ces dernieres "
        "annees, notamment dans le domaine du traitement du langage naturel. Les modeles "
        "de synthese vocale modernes sont capables de produire une parole extremement "
        "naturelle et expressive, ouvrant de nouvelles perspectives pour la communication.",
    ),
]


def run_timing_suite(base_url: str, output_dir: str, tries: int = 1):
    """Run timing benchmark: streaming vs batch on various text lengths, multiple tries."""
    print("\n" + "=" * 60)
    print(f"TIMING BENCHMARK SUITE ({tries} tries per combination)")
    print("=" * 60)

    # Collect all results for the summary table
    # Key: (label, mode) -> list of result dicts
    all_results: dict[tuple[str, str], list[dict]] = {}

    for label, text in TIMING_TEXTS:
        wc = len(text.split())
        print(f"\n--- {label} ({wc} words) ---")
        payload = build_payload(text=text, instruct=PARAM_INSTRUCT)

        for mode, fn in [("stream", synthesize_streaming), ("batch", synthesize_batch)]:
            key = (label, mode)
            all_results[key] = []

            for t in range(1, tries + 1):
                tag = f"{label}_{mode}_t{t}"
                print(f"  [{mode} #{t}] ", end="", flush=True)

                result = fn(base_url, payload)
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                    continue

                wav_path = os.path.join(output_dir, f"timing_{tag}.wav")
                save_wav(result["pcm_bytes"], wav_path)
                all_results[key].append(result)

                print(f"TTFB={result['ttfb_ms']:>5.0f}ms  "
                      f"total={result['total_ms']:>6.0f}ms  "
                      f"audio={result['audio_duration_s']:.1f}s  "
                      f"RT={result['realtime_factor']:.2f}x")

    # Summary table with averages
    rows = []
    for label, _text in TIMING_TEXTS:
        for mode in ("stream", "batch"):
            results = all_results.get((label, mode), [])
            if not results:
                rows.append([label, mode, "-", "-", "-", "-", "-"])
                continue
            avg_ttfb = sum(r["ttfb_ms"] for r in results) / len(results)
            avg_total = sum(r["total_ms"] for r in results) / len(results)
            avg_audio = sum(r["audio_duration_s"] for r in results) / len(results)
            avg_rt = sum(r["realtime_factor"] for r in results) / len(results)
            avg_pkts = sum(r["packets"] for r in results) / len(results)
            rows.append([
                label, mode,
                f"{avg_ttfb:.0f}",
                f"{avg_total:.0f}",
                f"{avg_audio:.1f}",
                f"{avg_rt:.2f}",
                f"{avg_pkts:.0f}",
            ])

    print(f"\n--- Average over {tries} tries ---")
    print_table(
        ["Text", "Mode", "TTFB(ms)", "Total(ms)", "Audio(s)", "RT Factor", "Packets"],
        rows,
    )


def run_custom_test(base_url: str, output_dir: str, args):
    """Run a single custom synthesis test."""
    print("\n" + "=" * 60)
    print("CUSTOM TEST")
    print("=" * 60)

    payload = build_payload(
        text=args.text,
        instruct=args.instruct,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"  Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    if args.instruct:
        print(f"  Instruct: {args.instruct}")
    print(f"  Params: temp={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} rep_pen={args.repetition_penalty}")

    use_streaming = not args.no_stream
    mode = "stream" if use_streaming else "batch"
    print(f"  Mode: {mode}")
    print()

    if use_streaming:
        result = synthesize_streaming(base_url, payload)
    else:
        result = synthesize_batch(base_url, payload)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    wav_path = os.path.join(output_dir, "custom.wav")
    save_wav(result["pcm_bytes"], wav_path)

    print(f"  TTFB:           {result['ttfb_ms']:.0f} ms")
    print(f"  Total time:     {result['total_ms']:.0f} ms")
    print(f"  Audio duration: {result['audio_duration_s']:.2f} s")
    print(f"  Realtime factor:{result['realtime_factor']:.2f}x")
    if use_streaming:
        print(f"  Packets:        {result['packets']}")
    print(f"  Saved to:       {wav_path}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Qwen3-TTS server with the Thalya voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--suite", choices=["styles", "params", "timing"],
        help="Run a specific test suite (default: run all)",
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Run a single custom test (use with --text, --instruct, etc.)",
    )
    parser.add_argument("--text", default="Bonjour, je suis Thalya. Comment allez-vous?")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Use batch endpoint instead of streaming (applies to all suites except timing)",
    )
    parser.add_argument(
        "--tries", type=int, default=4,
        help="Number of tries per combination in the timing suite (default: 4)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for WAV files (default: test_results/thalya_YYYYMMDD_HHMMSS/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.url.rstrip("/")

    output_dir = make_output_dir(args.output)

    print(f"Qwen3-TTS Thalya Test Script")
    print(f"Server: {base_url}")
    print(f"Output: {output_dir}/")
    print()

    # Pre-flight checks
    if not check_server(base_url):
        sys.exit(1)

    ensure_thalya_loaded(base_url)

    stream = not args.no_stream

    if args.custom:
        run_custom_test(base_url, output_dir, args)
    elif args.suite == "styles":
        run_styles_suite(base_url, output_dir, stream=stream)
    elif args.suite == "params":
        run_params_suite(base_url, output_dir, stream=stream)
    elif args.suite == "timing":
        run_timing_suite(base_url, output_dir, tries=args.tries)
    else:
        # Run all suites
        run_styles_suite(base_url, output_dir, stream=stream)
        run_params_suite(base_url, output_dir, stream=stream)
        run_timing_suite(base_url, output_dir, tries=args.tries)

    print(f"Done. Results in: {output_dir}/")
    print(f"  To clear: rm -rf {output_dir}")
    print(f"  To clear all runs: rm -rf {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
