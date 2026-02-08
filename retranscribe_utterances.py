"""
Re-transcribe all utterance files using Whisper.

This script fixes the text-audio misalignment issue where transcripts were
incorrectly extracted from segment boundaries of the full podcast transcription,
instead of running Whisper on each individual utterance file.
"""

import whisper
import os
from pathlib import Path
from tqdm import tqdm

TRAINING_DIR = Path("training_data")
TRANSCRIPTS_FILE = Path("transcripts.txt")


def get_utterance_files():
    """Get all utt*.wav files sorted numerically."""
    files = list(TRAINING_DIR.glob("utt*.wav"))
    # Sort numerically by extracting the number from uttXXX.wav
    files.sort(key=lambda f: int(f.stem[3:]))
    return files


def transcribe_utterances(model_name="large-v3"):
    """Transcribe all utterance files with Whisper."""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    files = get_utterance_files()
    print(f"Found {len(files)} utterance files")

    transcripts = []

    # Also add ref.wav entry at the beginning (if it exists and is long enough)
    ref_path = TRAINING_DIR / "ref.wav"
    if ref_path.exists():
        print(f"\nTranscribing ref.wav...")
        result = model.transcribe(str(ref_path), language="fr")
        ref_text = result["text"].strip()
        transcripts.append(("ref.wav", ref_text))
        print(f"  ref.wav: {ref_text[:80]}...")

    print(f"\nTranscribing {len(files)} utterances...")
    for audio_file in tqdm(files, desc="Transcribing"):
        result = model.transcribe(str(audio_file), language="fr")
        text = result["text"].strip()
        transcripts.append((audio_file.name, text))

    return transcripts


def write_transcripts(transcripts):
    """Write transcripts to file."""
    with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
        for filename, text in transcripts:
            f.write(f"{filename}|{text}\n")

    print(f"\nWrote {len(transcripts)} transcripts to {TRANSCRIPTS_FILE}")


def verify_transcripts(transcripts, sample_count=5):
    """Print sample transcripts for verification."""
    print(f"\n{'='*60}")
    print(f"Sample transcripts (first {sample_count}):")
    print('='*60)

    for filename, text in transcripts[:sample_count]:
        print(f"\n{filename}:")
        print(f"  {text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-transcribe utterances with Whisper")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model to use (default: large-v3)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing transcripts, don't retranscribe")
    args = parser.parse_args()

    if args.verify_only:
        # Just read and display existing transcripts
        if TRANSCRIPTS_FILE.exists():
            lines = TRANSCRIPTS_FILE.read_text(encoding="utf-8").splitlines()
            transcripts = []
            for line in lines:
                if line.strip() and "|" in line:
                    filename, text = line.split("|", 1)
                    transcripts.append((filename.strip(), text.strip()))
            verify_transcripts(transcripts)
        else:
            print(f"No transcripts file found at {TRANSCRIPTS_FILE}")
    else:
        transcripts = transcribe_utterances(args.model)
        write_transcripts(transcripts)
        verify_transcripts(transcripts)

        print(f"\n{'='*60}")
        print("Next steps:")
        print("  1. python setup_finetuning.py  # Generate train_raw.jsonl")
        print("  2. python qwen3-tts-repo/finetuning/prepare_data.py \\")
        print("       --input_jsonl train_raw.jsonl \\")
        print("       --output_jsonl train_with_codes.jsonl")
        print('='*60)
