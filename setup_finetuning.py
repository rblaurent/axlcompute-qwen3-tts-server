"""
Helper script to prepare data for Qwen3-TTS fine-tuning.

Usage:
1. Place your audio files in ./training_data/
2. Create transcripts.txt with format: filename.wav|Transcript text
3. Run this script to generate train_raw.jsonl
4. Run prepare_data.py and sft_12hz.py
"""

import os
import json
import librosa
import soundfile as sf
from pathlib import Path

TRAINING_DIR = Path("training_data")
REF_AUDIO = TRAINING_DIR / "ref.wav"
TRANSCRIPTS_FILE = Path("transcripts.txt")
OUTPUT_JSONL = Path("train_raw.jsonl")
TARGET_SR = 24000


def convert_to_24khz(input_path: Path, output_path: Path) -> bool:
    """Convert audio to 24kHz mono WAV."""
    try:
        audio, sr = librosa.load(str(input_path), sr=TARGET_SR, mono=True)
        sf.write(str(output_path), audio, TARGET_SR)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def check_audio_format(audio_path: Path) -> dict:
    """Check if audio meets requirements."""
    try:
        info = sf.info(str(audio_path))
        return {
            "path": str(audio_path),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "valid": info.samplerate == 24000 and info.channels == 1
        }
    except Exception as e:
        return {"path": str(audio_path), "error": str(e), "valid": False}


def setup_training_directory():
    """Create training directory structure."""
    TRAINING_DIR.mkdir(exist_ok=True)
    print(f"Training directory: {TRAINING_DIR.absolute()}")

    # Create example transcripts file
    if not TRANSCRIPTS_FILE.exists():
        example = """# Format: filename.wav|Transcript text (one per line)
# Example:
# utt001.wav|Bonjour, comment allez-vous aujourd'hui?
# utt002.wav|Je suis vraiment content de vous voir.
"""
        TRANSCRIPTS_FILE.write_text(example, encoding="utf-8")
        print(f"Created example transcripts file: {TRANSCRIPTS_FILE}")


def parse_transcripts() -> list:
    """Parse transcripts.txt into list of (filename, text) tuples."""
    if not TRANSCRIPTS_FILE.exists():
        print(f"Error: {TRANSCRIPTS_FILE} not found")
        return []

    entries = []
    for line in TRANSCRIPTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            print(f"Warning: Invalid line format (missing |): {line}")
            continue
        filename, text = line.split("|", 1)
        entries.append((filename.strip(), text.strip()))

    return entries


def generate_jsonl():
    """Generate train_raw.jsonl from transcripts and audio files."""
    entries = parse_transcripts()
    if not entries:
        print("No transcripts found. Add entries to transcripts.txt first.")
        return

    # Check reference audio
    if not REF_AUDIO.exists():
        print(f"Error: Reference audio not found: {REF_AUDIO}")
        print("Please add a 3+ second reference audio file as ref.wav")
        return

    ref_info = check_audio_format(REF_AUDIO)
    if not ref_info["valid"]:
        print(f"Warning: ref.wav is not 24kHz mono. Converting...")
        backup = REF_AUDIO.with_suffix(".original.wav")
        REF_AUDIO.rename(backup)
        convert_to_24khz(backup, REF_AUDIO)

    print(f"Reference audio: {REF_AUDIO} ({ref_info.get('duration', 0):.1f}s)")

    # Process each utterance
    jsonl_entries = []
    for filename, text in entries:
        audio_path = TRAINING_DIR / filename
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        # Check format
        info = check_audio_format(audio_path)
        if not info["valid"]:
            print(f"Converting {filename} to 24kHz mono...")
            backup = audio_path.with_suffix(".original.wav")
            audio_path.rename(backup)
            convert_to_24khz(backup, audio_path)

        entry = {
            "audio": f"./{audio_path}",
            "text": text,
            "ref_audio": f"./{REF_AUDIO}"
        }
        jsonl_entries.append(entry)
        print(f"  Added: {filename} ({info.get('duration', 0):.1f}s)")

    # Write JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nGenerated: {OUTPUT_JSONL} with {len(jsonl_entries)} entries")
    print("\nNext steps:")
    print("  1. python qwen3-tts-repo/finetuning/prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl")
    print("  2. python qwen3-tts-repo/finetuning/sft_12hz.py --train_jsonl train_with_codes.jsonl --speaker_name your_name")


def check_existing_audio():
    """List existing audio files in training directory."""
    if not TRAINING_DIR.exists():
        return

    wav_files = list(TRAINING_DIR.glob("*.wav"))
    if wav_files:
        print(f"\nExisting audio files in {TRAINING_DIR}:")
        for f in wav_files:
            info = check_audio_format(f)
            status = "OK" if info["valid"] else f"NEEDS CONVERSION ({info.get('samplerate')}Hz)"
            duration = f"{info.get('duration', 0):.1f}s" if "duration" in info else "error"
            print(f"  {f.name}: {duration} {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-TTS Fine-Tuning Data Setup")
    print("=" * 60)

    setup_training_directory()
    check_existing_audio()

    print("\n" + "=" * 60)
    print("Instructions:")
    print("=" * 60)
    print("""
1. Place your voice recordings in: training_data/
   - ref.wav: Reference audio (3+ seconds, clear speech)
   - utt001.wav, utt002.wav, ...: Training utterances

2. Edit transcripts.txt with format:
   filename.wav|Transcript text

3. Run: python setup_finetuning.py
   This will generate train_raw.jsonl

4. Prepare data (adds audio codes):
   python qwen3-tts-repo/finetuning/prepare_data.py \\
     --input_jsonl train_raw.jsonl \\
     --output_jsonl train_with_codes.jsonl

5. Fine-tune:
   python qwen3-tts-repo/finetuning/sft_12hz.py \\
     --train_jsonl train_with_codes.jsonl \\
     --speaker_name your_speaker_name \\
     --num_epochs 10

6. Test with style instructions:
   from qwen_tts import Qwen3TTSModel
   model = Qwen3TTSModel.from_pretrained("output/checkpoint-epoch-9", ...)
   wavs, sr = model.generate_custom_voice(
       text="Your text here",
       speaker="your_speaker_name",
       instruct="Style instruction here"
   )
""")

    # Check if transcripts exist and have entries
    if TRANSCRIPTS_FILE.exists():
        entries = parse_transcripts()
        if entries:
            print("\nFound transcripts. Generating JSONL...")
            generate_jsonl()
