"""
Fix harsh cuts at the beginning and end of utterances.

Adds silence padding and applies fade in/out to smooth transitions.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import shutil

TRAINING_DIR = Path("training_data")
BACKUP_DIR = Path("training_data_backup")
SAMPLE_RATE = 24000

# Padding and fade settings
SILENCE_PAD_MS = 150  # Add 150ms silence at start and end
FADE_MS = 30  # 30ms fade in/out


def add_padding_and_fade(audio, sr):
    """Add silence padding and fade in/out to audio."""
    silence_samples = int(SILENCE_PAD_MS / 1000 * sr)
    fade_samples = int(FADE_MS / 1000 * sr)

    # Create silence padding
    silence = np.zeros(silence_samples, dtype=audio.dtype)

    # Apply fade in to start of audio
    if fade_samples > 0 and len(audio) > fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in

    # Apply fade out to end of audio
    if fade_samples > 0 and len(audio) > fade_samples:
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out

    # Concatenate: silence + audio + silence
    padded = np.concatenate([silence, audio, silence])

    return padded


def process_utterances():
    """Process all utterance files."""
    files = sorted(TRAINING_DIR.glob("utt*.wav"), key=lambda f: int(f.stem[3:]))

    # Create backup
    if not BACKUP_DIR.exists():
        print(f"Creating backup in {BACKUP_DIR}...")
        BACKUP_DIR.mkdir(exist_ok=True)
        for f in files:
            shutil.copy2(f, BACKUP_DIR / f.name)
        print(f"Backed up {len(files)} files")
    else:
        print(f"Backup already exists at {BACKUP_DIR}")

    print(f"\nProcessing {len(files)} utterances...")
    print(f"  Adding {SILENCE_PAD_MS}ms silence padding")
    print(f"  Applying {FADE_MS}ms fade in/out")

    for f in tqdm(files, desc="Processing"):
        audio, sr = sf.read(f)

        # Skip if too short
        if len(audio) < sr * 0.5:  # Skip files shorter than 0.5s
            continue

        # Add padding and fade
        processed = add_padding_and_fade(audio, sr)

        # Save back
        sf.write(f, processed, sr)

    print("\nDone! Utterances now have proper silence boundaries.")
    print("\nNext steps:")
    print("  1. Re-run prepare_data.py to regenerate audio codes")
    print("  2. Re-run fine-tuning")


def analyze_boundaries():
    """Analyze current boundary status."""
    files = sorted(TRAINING_DIR.glob("utt*.wav"), key=lambda f: int(f.stem[3:]))

    harsh_start = 0
    harsh_end = 0

    for f in files:
        audio, sr = sf.read(f)
        window = int(0.05 * sr)  # 50ms

        if len(audio) < window * 2:
            continue

        start_rms = np.sqrt(np.mean(audio[:window]**2))
        end_rms = np.sqrt(np.mean(audio[-window:]**2))

        if start_rms > 0.02:
            harsh_start += 1
        if end_rms > 0.02:
            harsh_end += 1

    total = len(files)
    print(f"Boundary analysis:")
    print(f"  Harsh starts: {harsh_start}/{total} ({100*harsh_start/total:.1f}%)")
    print(f"  Harsh ends: {harsh_end}/{total} ({100*harsh_end/total:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix utterance boundaries")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't modify")
    parser.add_argument("--pad-ms", type=int, default=SILENCE_PAD_MS, help="Silence padding in ms")
    parser.add_argument("--fade-ms", type=int, default=FADE_MS, help="Fade duration in ms")
    args = parser.parse_args()

    if args.analyze:
        analyze_boundaries()
    else:
        SILENCE_PAD_MS = args.pad_ms
        FADE_MS = args.fade_ms
        process_utterances()
        print("\nAfter processing:")
        analyze_boundaries()
