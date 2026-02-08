#!/usr/bin/env python3
"""
Parameter sweep test for Qwen3-TTS with Thalya voice.

Runs 10 iterations of each parameter combination on a fixed French sentence
with "Teasing" style instruction, to find the best generation settings.

Sweeps: temperature, top_k, top_p, repetition_penalty,
        subtalker_temperature, subtalker_top_k, subtalker_top_p

Usage:
  python test_thalya_param_sweep.py                    # Run all sweeps
  python test_thalya_param_sweep.py --sweep temperature # Single sweep
  python test_thalya_param_sweep.py --iterations 5      # Fewer iterations
  python test_thalya_param_sweep.py --url http://host:port
"""

import argparse
import csv
import datetime
import math
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

TEST_TEXT = (
    "Ah mais attends, t'as genre trois terminaux ouverts en même temps là? "
    "C'est quoi cette organisation chaotique, sérieux?"
)
TEST_INSTRUCT = "Teasing"

ITERATIONS = 10

# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------
SWEEPS = {
    "temperature": {
        "param": "temperature",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5],
    },
    "top_k": {
        "param": "top_k",
        "values": [5, 10, 20, 30, 50, 80, 100, 150],
    },
    "top_p": {
        "param": "top_p",
        "values": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    },
    "repetition_penalty": {
        "param": "repetition_penalty",
        "values": [1.0, 1.02, 1.05, 1.1, 1.15, 1.2],
    },
    "subtalker_temperature": {
        "param": "subtalker_temperature",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2],
    },
    "subtalker_top_k": {
        "param": "subtalker_top_k",
        "values": [5, 10, 20, 30, 50, 80, 100],
    },
    "subtalker_top_p": {
        "param": "subtalker_top_p",
        "values": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dir(base: str | None = None) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base or os.path.join(OUTPUT_ROOT, f"param_sweep_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def save_wav(pcm_bytes: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def audio_duration_s(pcm_bytes: bytes) -> float:
    return (len(pcm_bytes) // 2) / SAMPLE_RATE


def check_server(base_url: str) -> bool:
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
    try:
        r = requests.get(f"{base_url}/model/info", timeout=5)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        print(f"Failed to get model info: {e}")
        sys.exit(1)

    model_path = info.get("model_path", "") or ""
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


def build_payload(**overrides) -> dict:
    payload = {
        "text": TEST_TEXT,
        "speaker": SPEAKER,
        "language": LANGUAGE,
        "instruct": TEST_INSTRUCT,
    }
    for k, v in overrides.items():
        if v is not None:
            payload[k] = v
    return payload


def synthesize_streaming(base_url: str, payload: dict) -> dict:
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


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
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
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep(
    base_url: str,
    output_dir: str,
    sweep_name: str,
    param_name: str,
    values: list,
    iterations: int,
) -> list[dict]:
    """Run one parameter sweep. Returns list of summary dicts for CSV export."""
    print(f"\n{'=' * 70}")
    print(f"SWEEP: {param_name}  |  {iterations} iterations x {len(values)} values")
    print(f"Text: \"{TEST_TEXT[:70]}...\"")
    print(f"Style: \"{TEST_INSTRUCT}\"")
    print(f"{'=' * 70}")

    sweep_dir = os.path.join(output_dir, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    summary_rows = []  # for table display
    csv_rows = []      # for CSV export

    for val in values:
        label = f"{param_name}={val}"
        print(f"\n  --- {label} ---")

        ttfbs = []
        totals = []
        durations = []
        rtfs = []
        errors = 0

        for i in range(1, iterations + 1):
            print(f"    [{i:2d}/{iterations}] ", end="", flush=True)

            payload = build_payload(**{param_name: val})
            result = synthesize_streaming(base_url, payload)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                errors += 1
                continue

            wav_path = os.path.join(sweep_dir, f"{param_name}_{val}_iter{i:02d}.wav")
            save_wav(result["pcm_bytes"], wav_path)

            ttfbs.append(result["ttfb_ms"])
            totals.append(result["total_ms"])
            durations.append(result["audio_duration_s"])
            rtfs.append(result["realtime_factor"])

            print(
                f"TTFB={result['ttfb_ms']:>5.0f}ms  "
                f"total={result['total_ms']:>6.0f}ms  "
                f"audio={result['audio_duration_s']:.2f}s  "
                f"RT={result['realtime_factor']:.3f}x"
            )

        # Compute stats
        n = len(ttfbs)
        if n == 0:
            summary_rows.append([str(val), "ALL ERRORS", "-", "-", "-", "-", "-"])
            continue

        avg_ttfb = sum(ttfbs) / n
        avg_total = sum(totals) / n
        avg_dur = sum(durations) / n
        avg_rtf = sum(rtfs) / n
        sd_ttfb = stddev(ttfbs)
        sd_total = stddev(totals)
        sd_dur = stddev(durations)
        sd_rtf = stddev(rtfs)

        summary_rows.append([
            str(val),
            f"{n}/{iterations}",
            f"{avg_ttfb:.0f} +/- {sd_ttfb:.0f}",
            f"{avg_total:.0f} +/- {sd_total:.0f}",
            f"{avg_dur:.2f} +/- {sd_dur:.2f}",
            f"{avg_rtf:.3f} +/- {sd_rtf:.3f}",
        ])

        csv_rows.append({
            "param": param_name,
            "value": val,
            "n": n,
            "errors": errors,
            "avg_ttfb_ms": round(avg_ttfb, 1),
            "sd_ttfb_ms": round(sd_ttfb, 1),
            "avg_total_ms": round(avg_total, 1),
            "sd_total_ms": round(sd_total, 1),
            "avg_audio_s": round(avg_dur, 3),
            "sd_audio_s": round(sd_dur, 3),
            "avg_rtf": round(avg_rtf, 4),
            "sd_rtf": round(sd_rtf, 4),
            "min_rtf": round(min(rtfs), 4),
            "max_rtf": round(max(rtfs), 4),
            "min_audio_s": round(min(durations), 3),
            "max_audio_s": round(max(durations), 3),
        })

    # Print summary table
    print(f"\n--- {param_name} Summary (mean +/- stddev) ---")
    print_table(
        [param_name, "OK", "TTFB(ms)", "Total(ms)", "Audio(s)", "RT Factor"],
        summary_rows,
    )

    return csv_rows


def write_csv(csv_rows: list[dict], path: str):
    if not csv_rows:
        return
    fieldnames = list(csv_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV results saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter sweep test for Qwen3-TTS Thalya voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--sweep", nargs="*", choices=list(SWEEPS.keys()),
        help="Run specific sweep(s). Default: run all.",
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS,
        help=f"Iterations per parameter value (default: {ITERATIONS})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: test_results/param_sweep_YYYYMMDD_HHMMSS/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.url.rstrip("/")
    output_dir = make_output_dir(args.output)
    iterations = args.iterations

    sweep_names = args.sweep if args.sweep else list(SWEEPS.keys())

    print("Qwen3-TTS Parameter Sweep Test")
    print(f"Server:     {base_url}")
    print(f"Output:     {output_dir}/")
    print(f"Iterations: {iterations}")
    print(f"Sweeps:     {', '.join(sweep_names)}")
    print(f"Text:       \"{TEST_TEXT}\"")
    print(f"Style:      \"{TEST_INSTRUCT}\"")
    print()

    if not check_server(base_url):
        sys.exit(1)

    ensure_thalya_loaded(base_url)

    all_csv_rows = []

    for name in sweep_names:
        sweep_def = SWEEPS[name]
        rows = run_sweep(
            base_url=base_url,
            output_dir=output_dir,
            sweep_name=name,
            param_name=sweep_def["param"],
            values=sweep_def["values"],
            iterations=iterations,
        )
        all_csv_rows.extend(rows)

    # Write combined CSV
    csv_path = os.path.join(output_dir, "param_sweep_results.csv")
    write_csv(all_csv_rows, csv_path)

    print(f"\nDone. All results in: {output_dir}/")
    print(f"  WAV files in per-sweep subdirectories")
    print(f"  Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
