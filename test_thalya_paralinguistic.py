#!/usr/bin/env python3
"""
Paralinguistic sound exploration for Qwen3-TTS with Thalya voice.

Systematically tests strategies to coax actual laughter and sighing sounds
out of the model by varying THREE axes simultaneously:
  1. Text content (plain vs onomatopoeia)
  2. Instruct wording (none, English, descriptive, Chinese)
  3. Generation parameters (temperature, top_k, top_p, subtalker_temperature)

Grouped into curated experiment sets that test specific hypotheses:
  - H1: Does onomatopoeia in text produce actual laughter/sigh sounds?
  - H2: Does high temperature unlock paralinguistic sounds?
  - H3: Does the full combo (text + instruct + heat) work?

Usage:
  python test_thalya_paralinguistic.py
  python test_thalya_paralinguistic.py --group laugh_text_effect
  python test_thalya_paralinguistic.py --group laugh_temperature sigh_temperature
  python test_thalya_paralinguistic.py --iterations 5
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
ITERATIONS = 3

# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------

# Plain texts (no onomatopoeia) -- baseline
LAUGH_TEXT_PLAIN = (
    "Non mais j'y crois pas! T'as vraiment fait \u00e7a? "
    "Devant tout le monde en plus!"
)
SIGH_TEXT_PLAIN = (
    "Ah mais attends, t'as genre trois terminaux ouverts en m\u00eame temps l\u00e0? "
    "C'est quoi cette organisation chaotique, s\u00e9rieux?"
)

# Onomatopoeia embedded in text
LAUGH_TEXT_HAHA = "Ha ha ha ha! Non mais s\u00e9rieux, ha ha! C'est trop dr\u00f4le \u00e7a!"
LAUGH_TEXT_HIHI = "Hi hi hi! Oh non, j'y crois pas! Hi hi!"
LAUGH_TEXT_MIXED = (
    "Ha ha ha ha! Non mais t'es pas s\u00e9rieux? Ha ha! "
    "Oh j'en peux plus, ha ha ha!"
)

SIGH_TEXT_PFFF = "Pfff... bon, c'est quoi cette organisation chaotique, s\u00e9rieux?"
SIGH_TEXT_AAH = "Aaah... non mais attends, j'en ai marre de tout \u00e7a."
SIGH_TEXT_MIXED = "Pfff... aaah, c'est pas possible. Bon... on fait quoi maintenant?"

# Tag -> text mapping for display
TEXT_LOOKUP = {
    "plain_laugh": LAUGH_TEXT_PLAIN,
    "haha": LAUGH_TEXT_HAHA,
    "hihi": LAUGH_TEXT_HIHI,
    "mixed_laugh": LAUGH_TEXT_MIXED,
    "plain_sigh": SIGH_TEXT_PLAIN,
    "pfff": SIGH_TEXT_PFFF,
    "aah": SIGH_TEXT_AAH,
    "mixed_sigh": SIGH_TEXT_MIXED,
}

# ---------------------------------------------------------------------------
# Instructs
# ---------------------------------------------------------------------------
LAUGH_INSTRUCTS = [
    ("no_instruct", None),
    ("laughing", "Laughing"),
    ("laughing_hard", "Laughing hard, barely able to speak"),
    ("giggling_between_words", "Giggling between words"),
    ("burst_out_laughing", "Burst out laughing while speaking"),
    ("cant_stop_laughing", "Can't stop laughing, losing composure"),
    ("zh_laughing", "\u54c8\u54c8\u5927\u7b11\u7740\u8bf4"),
]

SIGH_INSTRUCTS = [
    ("no_instruct", None),
    ("sighing", "Sighing"),
    ("heavy_sigh", "Heavy sigh before speaking"),
    ("exhausted_sigh", "Exhausted, sighing deeply"),
    ("exasperated_breath", "Let out an exasperated breath"),
    ("deep_breath_then_speak", "Take a deep breath, then speak tiredly"),
    ("zh_sighing", "\u53f9\u4e86\u53e3\u6c14\u8bf4"),
]

# ---------------------------------------------------------------------------
# Generation parameter sets
# ---------------------------------------------------------------------------
GEN_PARAMS = {
    "default":   {},  # server defaults (temp=0.9, top_k=50, top_p=1.0)
    "hot":       {"temperature": 1.2, "top_p": 0.95},
    "very_hot":  {"temperature": 1.5, "top_p": 0.98},
    "hot_lowk":  {"temperature": 1.2, "top_k": 20, "top_p": 0.9},
    "sub_hot":   {"subtalker_temperature": 1.3},
    "both_hot":  {"temperature": 1.2, "subtalker_temperature": 1.3, "top_p": 0.95},
}

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------
EXPERIMENT_GROUPS = {
    # H1: Does onomatopoeia in text produce actual laughter sounds?
    "laugh_text_effect": {
        "texts": [
            ("plain_laugh", LAUGH_TEXT_PLAIN),
            ("haha", LAUGH_TEXT_HAHA),
            ("hihi", LAUGH_TEXT_HIHI),
            ("mixed_laugh", LAUGH_TEXT_MIXED),
        ],
        "instructs": [
            ("no_instruct", None),
            ("laughing", "Laughing"),
        ],
        "params": [
            ("default", {}),
        ],
    },

    # H2: Does high temperature unlock laughter?
    "laugh_temperature": {
        "texts": [
            ("haha", LAUGH_TEXT_HAHA),
        ],
        "instructs": [
            ("laughing", "Laughing"),
            ("burst_out_laughing", "Burst out laughing while speaking"),
        ],
        "params": list(GEN_PARAMS.items()),
    },

    # H3: Does descriptive instruct + onomatopoeia + heat = laughter?
    "laugh_full_combo": {
        "texts": [
            ("haha", LAUGH_TEXT_HAHA),
            ("mixed_laugh", LAUGH_TEXT_MIXED),
        ],
        "instructs": LAUGH_INSTRUCTS,
        "params": [
            ("default", {}),
            ("hot", GEN_PARAMS["hot"]),
            ("very_hot", GEN_PARAMS["very_hot"]),
        ],
    },

    # H4: Does onomatopoeia in text produce actual sigh sounds?
    "sigh_text_effect": {
        "texts": [
            ("plain_sigh", SIGH_TEXT_PLAIN),
            ("pfff", SIGH_TEXT_PFFF),
            ("aah", SIGH_TEXT_AAH),
            ("mixed_sigh", SIGH_TEXT_MIXED),
        ],
        "instructs": [
            ("no_instruct", None),
            ("sighing", "Sighing"),
        ],
        "params": [
            ("default", {}),
        ],
    },

    # H5: Does high temperature unlock sighing?
    "sigh_temperature": {
        "texts": [
            ("pfff", SIGH_TEXT_PFFF),
        ],
        "instructs": [
            ("sighing", "Sighing"),
            ("heavy_sigh", "Heavy sigh before speaking"),
        ],
        "params": list(GEN_PARAMS.items()),
    },

    # H6: Full combo for sighing
    "sigh_full_combo": {
        "texts": [
            ("pfff", SIGH_TEXT_PFFF),
            ("mixed_sigh", SIGH_TEXT_MIXED),
        ],
        "instructs": SIGH_INSTRUCTS,
        "params": [
            ("default", {}),
            ("hot", GEN_PARAMS["hot"]),
            ("very_hot", GEN_PARAMS["very_hot"]),
        ],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dir(base: str | None = None) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base or os.path.join(OUTPUT_ROOT, f"paralinguistic_{stamp}")
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


def write_csv(csv_rows: list[dict], path: str):
    if not csv_rows:
        return
    fieldnames = list(csv_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV results saved to: {path}")


def format_params(params: dict) -> str:
    if not params:
        return "(server defaults)"
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def generate_combinations(group_def: dict) -> list[dict]:
    """Generate all (text, instruct, params) combinations for a group."""
    combos = []
    for text_tag, text in group_def["texts"]:
        for instruct_tag, instruct in group_def["instructs"]:
            for param_tag, params in group_def["params"]:
                combos.append({
                    "text_tag": text_tag,
                    "text": text,
                    "instruct_tag": instruct_tag,
                    "instruct": instruct,
                    "param_tag": param_tag,
                    "params": params,
                })
    return combos


def run_experiment_group(
    base_url: str,
    output_dir: str,
    group_name: str,
    group_def: dict,
    iterations: int,
) -> list[dict]:
    combos = generate_combinations(group_def)

    print(f"\n{'=' * 78}")
    print(f"GROUP: {group_name}  |  {len(combos)} combos x {iterations} iters = {len(combos) * iterations} runs")
    print(f"  Texts:     {len(group_def['texts'])}")
    print(f"  Instructs: {len(group_def['instructs'])}")
    print(f"  Params:    {len(group_def['params'])}")
    print(f"{'=' * 78}")

    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    summary_rows = []
    csv_rows = []

    for ci, combo in enumerate(combos, 1):
        text_tag = combo["text_tag"]
        text = combo["text"]
        instruct_tag = combo["instruct_tag"]
        instruct = combo["instruct"]
        param_tag = combo["param_tag"]
        params = combo["params"]

        combo_label = f"{text_tag}+{instruct_tag}+{param_tag}"
        instruct_display = repr(instruct) if instruct is not None else "None"
        if len(instruct_display) > 40:
            instruct_display = instruct_display[:37] + "..."

        print(f"\n  [{ci}/{len(combos)}] {combo_label}")
        print(f"    text: \"{text[:60]}...\"" if len(text) > 60 else f"    text: \"{text}\"")
        print(f"    instruct: {instruct_display}")
        print(f"    params: {format_params(params)}")

        ttfbs = []
        totals = []
        durations = []
        rtfs = []
        errors = 0

        for i in range(1, iterations + 1):
            print(f"    iter [{i:2d}/{iterations}] ", end="", flush=True)

            payload = {
                "text": text,
                "speaker": SPEAKER,
                "language": LANGUAGE,
            }
            if instruct is not None:
                payload["instruct"] = instruct
            payload.update(params)

            result = synthesize_streaming(base_url, payload)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                errors += 1
                continue

            wav_name = f"{text_tag}_{instruct_tag}_{param_tag}_iter{i:02d}.wav"
            wav_path = os.path.join(group_dir, wav_name)
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
            summary_rows.append([
                combo_label, "ALL ERR", "-", "-", "-", "-",
            ])
            continue

        avg_dur = sum(durations) / n
        sd_dur = stddev(durations)
        avg_rtf = sum(rtfs) / n
        avg_ttfb = sum(ttfbs) / n
        avg_total = sum(totals) / n

        summary_rows.append([
            combo_label,
            f"{n}/{iterations}",
            f"{avg_ttfb:.0f}",
            f"{avg_total:.0f}",
            f"{avg_dur:.2f} +/- {sd_dur:.2f}",
            f"{avg_rtf:.3f}",
        ])

        csv_rows.append({
            "group": group_name,
            "text_tag": text_tag,
            "text": text,
            "instruct_tag": instruct_tag,
            "instruct": instruct if instruct is not None else "<None>",
            "param_tag": param_tag,
            "temperature": params.get("temperature", ""),
            "top_k": params.get("top_k", ""),
            "top_p": params.get("top_p", ""),
            "subtalker_temperature": params.get("subtalker_temperature", ""),
            "n": n,
            "errors": errors,
            "avg_ttfb_ms": round(avg_ttfb, 1),
            "avg_total_ms": round(avg_total, 1),
            "avg_audio_s": round(avg_dur, 3),
            "sd_audio_s": round(sd_dur, 3),
            "min_audio_s": round(min(durations), 3),
            "max_audio_s": round(max(durations), 3),
            "avg_rtf": round(avg_rtf, 4),
        })

    print(f"\n--- {group_name} Summary ---")
    print_table(
        ["Combo", "OK", "TTFB(ms)", "Total(ms)", "Audio(s)", "RT Factor"],
        summary_rows,
    )

    return csv_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paralinguistic sound exploration for Qwen3-TTS Thalya voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--group", nargs="*", choices=list(EXPERIMENT_GROUPS.keys()),
        help="Run specific group(s). Default: run all.",
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS,
        help=f"Iterations per combination (default: {ITERATIONS})",
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

    group_names = args.group if args.group else list(EXPERIMENT_GROUPS.keys())

    total_combos = 0
    for g in group_names:
        gd = EXPERIMENT_GROUPS[g]
        total_combos += len(gd["texts"]) * len(gd["instructs"]) * len(gd["params"])
    total_runs = total_combos * iterations

    print("Qwen3-TTS Paralinguistic Sound Exploration")
    print(f"Server:     {base_url}")
    print(f"Output:     {output_dir}/")
    print(f"Iterations: {iterations}")
    print(f"Groups:     {', '.join(group_names)}")
    print(f"Total:      {total_combos} combos x {iterations} iters = {total_runs} runs")
    print()

    # Show texts being tested
    texts_used = set()
    for g in group_names:
        for tag, _ in EXPERIMENT_GROUPS[g]["texts"]:
            texts_used.add(tag)
    for tag in sorted(texts_used):
        t = TEXT_LOOKUP[tag]
        display = f"\"{t[:65]}...\"" if len(t) > 65 else f"\"{t}\""
        print(f"  [{tag}] {display}")
    print()

    if not check_server(base_url):
        sys.exit(1)

    ensure_thalya_loaded(base_url)

    all_csv_rows = []

    for name in group_names:
        rows = run_experiment_group(
            base_url=base_url,
            output_dir=output_dir,
            group_name=name,
            group_def=EXPERIMENT_GROUPS[name],
            iterations=iterations,
        )
        all_csv_rows.extend(rows)

    csv_path = os.path.join(output_dir, "paralinguistic_results.csv")
    write_csv(all_csv_rows, csv_path)

    print(f"\nDone. All results in: {output_dir}/")
    print(f"  WAV files in per-group subdirectories")
    print(f"  Summary CSV: {csv_path}")
    print()
    print("Next steps:")
    print("  1. Listen to WAVs -- compare plain text vs onomatopoeia")
    print("  2. Compare default temp vs hot/very_hot for paralinguistic sounds")
    print("  3. Check CSV audio durations (longer = model adding extra sounds?)")


if __name__ == "__main__":
    main()
