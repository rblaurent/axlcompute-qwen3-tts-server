#!/usr/bin/env python3
"""
Emotion & sound sweep for Qwen3-TTS with Thalya voice.

Tests how strong emotions and paralinguistic sounds (laughing, raging,
crying, whispering, sighing, etc.) affect voice quality/expressiveness.
Compares short-label vs descriptive instructions for voice preservation.

Usage:
  python test_thalya_emotion_sweep.py
  python test_thalya_emotion_sweep.py --iterations 2 --group laughing
  python test_thalya_emotion_sweep.py --group laughing anger sadness
"""

import argparse
import csv
import datetime
import io
import math
import os
import sys
import time
import wave

# Force UTF-8 output on Windows to handle emoji and accented chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

# Three test texts for different emotional contexts
TEST_TEXT_1 = (
    "Ah mais attends, t'as genre trois terminaux ouverts en même temps là? "
    "C'est quoi cette organisation chaotique, sérieux?"
)
TEST_TEXT_2 = (
    "Non mais j'y crois pas! T'as vraiment fait ça? "
    "Devant tout le monde en plus!"
)
TEST_TEXT_3 = (
    "Hé... t'inquiète pas, ça va aller. "
    "On va trouver une solution, je te promets."
)

ITERATIONS = 5

# ---------------------------------------------------------------------------
# Emotion/sound groups
# ---------------------------------------------------------------------------
EMOTION_GROUPS = {
    # --- LAUGHING / AMUSEMENT ---
    "laughing": [
        ("laughing", "Laughing"),
        ("amused", "Amused"),
        ("giggling", "Giggling"),
        ("cracking_up", "Cracking up"),
        ("stifled_laugh", "Stifled laugh"),
        ("laughing_while_speaking", "Laughing while speaking"),
        ("barely_containing_laughter", "Barely containing laughter"),
        ("laughing_then_composing", "Start laughing then compose yourself"),
    ],

    # --- ANGER / RAGE ---
    "anger": [
        ("angry", "Angry"),
        ("furious", "Furious"),
        ("raging", "Raging"),
        ("seething", "Seething"),
        ("cold_fury", "Cold fury"),
        ("explosive_anger", "Explosive anger"),
        ("frustrated_outburst", "Frustrated outburst"),
        ("angry_disbelief", "Angry disbelief"),
    ],

    # --- SADNESS / CRYING ---
    "sadness": [
        ("sad", "Sad"),
        ("tearful", "Tearful"),
        ("crying", "Crying"),
        ("sobbing", "Sobbing"),
        ("heartbroken", "Heartbroken"),
        ("choking_up", "Choking up"),
        ("sad_and_tearful", "Sad and tearful voice"),
        ("holding_back_tears", "Holding back tears"),
    ],

    # --- WHISPERING / QUIET ---
    "whisper": [
        ("whisper", "Whisper"),
        ("whispering", "Whispering"),
        ("hushed", "Hushed"),
        ("secretive", "Secretive"),
        ("soft_whisper", "Soft whisper"),
        ("conspiratorial_whisper", "Conspiratorial whisper"),
        ("excited_whisper", "Whisper excitedly as if sharing a secret"),
        ("loud_whisper", "Loud whisper"),
    ],

    # --- EXCITEMENT / HIGH ENERGY ---
    "excitement": [
        ("excited", "Excited"),
        ("ecstatic", "Ecstatic"),
        ("thrilled", "Thrilled"),
        ("hyped", "Hyped"),
        ("overjoyed", "Overjoyed"),
        ("giddy", "Giddy"),
        ("bursting_with_joy", "Bursting with joy"),
        ("can_barely_contain", "Can barely contain excitement"),
    ],

    # --- FEAR / ANXIETY ---
    "fear": [
        ("scared", "Scared"),
        ("terrified", "Terrified"),
        ("anxious", "Anxious"),
        ("panicked", "Panicked"),
        ("nervous", "Nervous"),
        ("trembling_fear", "Trembling with fear"),
        ("creeping_dread", "Creeping dread"),
        ("startled", "Startled"),
    ],

    # --- SIGHING / EXHAUSTION ---
    "sighing": [
        ("sighing", "Sighing"),
        ("exhausted", "Exhausted"),
        ("exasperated", "Exasperated"),
        ("weary", "Weary"),
        ("done_with_it", "So done with this"),
        ("heavy_sigh", "Heavy sigh"),
        ("resigned", "Resigned"),
        ("deep_breath", "Deep breath then speaking"),
    ],

    # --- DISGUST / CONTEMPT ---
    "disgust": [
        ("disgusted", "Disgusted"),
        ("revolted", "Revolted"),
        ("contemptuous", "Contemptuous"),
        ("sneering", "Sneering"),
        ("grossed_out", "Grossed out"),
        ("judgmental_disgust", "Judgmental disgust"),
    ],

    # --- MIXED / COMPLEX EMOTIONS ---
    "mixed_emotions": [
        ("laughing_through_tears", "Laughing through tears"),
        ("angry_but_amused", "Angry but amused"),
        ("scared_but_brave", "Scared but brave"),
        ("sad_smile", "Sad smile"),
        ("nervous_excitement", "Nervous excitement"),
        ("bittersweet", "Bittersweet"),
        ("incredulous_panic", "Speak in an incredulous tone, with a hint of panic"),
    ],
}

# Which test text fits each emotion group
TEXT_MAP = {
    "laughing": TEST_TEXT_1,
    "anger": TEST_TEXT_2,
    "sadness": TEST_TEXT_3,
    "whisper": TEST_TEXT_3,
    "excitement": TEST_TEXT_2,
    "fear": TEST_TEXT_2,
    "sighing": TEST_TEXT_1,
    "disgust": TEST_TEXT_2,
    "mixed_emotions": TEST_TEXT_2,
}

# Reverse lookup for CSV: which text ID was used
TEXT_ID_MAP = {
    "laughing": 1,
    "anger": 2,
    "sadness": 3,
    "whisper": 3,
    "excitement": 2,
    "fear": 2,
    "sighing": 1,
    "disgust": 2,
    "mixed_emotions": 2,
}

# ---------------------------------------------------------------------------
# Helpers (same as style sweep)
# ---------------------------------------------------------------------------

def make_output_dir(base: str | None = None) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base or os.path.join(OUTPUT_ROOT, f"emotion_sweep_{stamp}")
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
# Sweep logic
# ---------------------------------------------------------------------------

def run_emotion_group(
    base_url: str,
    output_dir: str,
    group_name: str,
    styles: list[tuple[str, str]],
    text: str,
    text_id: int,
    iterations: int,
) -> list[dict]:
    print(f"\n{'=' * 70}")
    print(f"GROUP: {group_name}  |  {iterations} iterations x {len(styles)} emotions")
    print(f"Text #{text_id}: \"{text[:70]}...\"")
    print(f"{'=' * 70}")

    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    summary_rows = []
    csv_rows = []

    for tag, instruct in styles:
        instruct_display = repr(instruct)
        if len(instruct_display) > 60:
            instruct_display = instruct_display[:57] + "..."
        print(f"\n  --- {tag}: {instruct_display} ---")

        ttfbs = []
        totals = []
        durations = []
        rtfs = []
        errors = 0

        for i in range(1, iterations + 1):
            print(f"    [{i:2d}/{iterations}] ", end="", flush=True)

            payload = {
                "text": text,
                "speaker": SPEAKER,
                "language": LANGUAGE,
                "instruct": instruct,
            }

            result = synthesize_streaming(base_url, payload)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                errors += 1
                continue

            wav_path = os.path.join(group_dir, f"{tag}_iter{i:02d}.wav")
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

        n = len(ttfbs)
        if n == 0:
            summary_rows.append([tag, instruct_display, "ALL ERR", "-", "-", "-"])
            continue

        avg_dur = sum(durations) / n
        sd_dur = stddev(durations)
        avg_rtf = sum(rtfs) / n
        avg_ttfb = sum(ttfbs) / n

        summary_rows.append([
            tag,
            instruct_display[:40],
            f"{n}/{iterations}",
            f"{avg_ttfb:.0f}",
            f"{avg_dur:.2f} +/- {sd_dur:.2f}",
            f"{avg_rtf:.3f}",
        ])

        csv_rows.append({
            "group": group_name,
            "tag": tag,
            "instruct": instruct,
            "text_id": text_id,
            "n": n,
            "errors": errors,
            "avg_ttfb_ms": round(avg_ttfb, 1),
            "avg_total_ms": round(sum(totals) / n, 1),
            "avg_audio_s": round(avg_dur, 3),
            "sd_audio_s": round(sd_dur, 3),
            "min_audio_s": round(min(durations), 3),
            "max_audio_s": round(max(durations), 3),
            "avg_rtf": round(avg_rtf, 4),
        })

    print(f"\n--- {group_name} Summary ---")
    print_table(
        ["Tag", "Instruct", "OK", "TTFB(ms)", "Audio(s)", "RT Factor"],
        summary_rows,
    )

    return csv_rows


def write_csv(csv_rows: list[dict], path: str):
    if not csv_rows:
        return
    fieldnames = list(csv_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV results saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Emotion & sound sweep for Qwen3-TTS Thalya voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--group", nargs="*", choices=list(EMOTION_GROUPS.keys()),
        help="Run specific group(s). Default: run all.",
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS,
        help=f"Iterations per emotion (default: {ITERATIONS})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.url.rstrip("/")
    output_dir = make_output_dir(args.output)
    iterations = args.iterations

    group_names = args.group if args.group else list(EMOTION_GROUPS.keys())

    total_emotions = sum(len(EMOTION_GROUPS[g]) for g in group_names)
    total_runs = total_emotions * iterations

    print("Qwen3-TTS Emotion & Sound Sweep")
    print(f"Server:     {base_url}")
    print(f"Output:     {output_dir}/")
    print(f"Iterations: {iterations}")
    print(f"Groups:     {', '.join(group_names)}")
    print(f"Total:      {total_emotions} emotions x {iterations} iters = {total_runs} runs")
    print()
    print(f"Text #1: \"{TEST_TEXT_1[:60]}...\"")
    print(f"Text #2: \"{TEST_TEXT_2[:60]}...\"")
    print(f"Text #3: \"{TEST_TEXT_3[:60]}...\"")
    print()

    if not check_server(base_url):
        sys.exit(1)

    ensure_thalya_loaded(base_url)

    all_csv_rows = []

    for name in group_names:
        rows = run_emotion_group(
            base_url=base_url,
            output_dir=output_dir,
            group_name=name,
            styles=EMOTION_GROUPS[name],
            text=TEXT_MAP[name],
            text_id=TEXT_ID_MAP[name],
            iterations=iterations,
        )
        all_csv_rows.extend(rows)

    csv_path = os.path.join(output_dir, "emotion_sweep_results.csv")
    write_csv(all_csv_rows, csv_path)

    print(f"\nDone. All results in: {output_dir}/")
    print(f"  WAV files in per-group subdirectories")
    print(f"  Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
