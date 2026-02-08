#!/usr/bin/env python3
"""
Single-folder voice training pipeline for Qwen3-TTS.

Usage:
    python train_voice.py <folder_path> [--epochs 5] [--lr 2e-6] [--batch_size 2]

Example:
    1. Create folder: mkdir thalya
    2. Add .ogg files: cp *.ogg thalya/
    3. Run: python train_voice.py thalya
    4. Done! Model saved in thalya/model/

The folder name becomes the speaker name.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Add qwen3-tts-repo to path
BASE_DIR = Path(__file__).parent
QWEN_REPO = BASE_DIR / "qwen3-tts-repo"
if str(QWEN_REPO) not in sys.path:
    sys.path.insert(0, str(QWEN_REPO))
if str(QWEN_REPO / "finetuning") not in sys.path:
    sys.path.insert(0, str(QWEN_REPO / "finetuning"))


def print_step(step_num: int, title: str):
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {title}")
    print(f"{'='*60}\n")


# ============================================================================
# Step 1: Convert OGG to WAV
# ============================================================================

def convert_audio_to_wav(folder: Path) -> list[Path]:
    """Convert all audio files in folder to 24kHz mono WAV."""
    print_step(1, "Converting Audio to WAV")

    from pydub import AudioSegment

    # Supported audio formats
    audio_extensions = ["*.ogg", "*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.wma"]

    # Find all audio files (exclude utterances subfolder)
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder.glob(ext))

    # Sort and filter out any files already in subfolders
    audio_files = sorted([f for f in audio_files if f.parent == folder])

    if not audio_files:
        print("  No audio files found!")
        print(f"  Supported formats: {', '.join(e.replace('*', '') for e in audio_extensions)}")
        return []

    print(f"  Found {len(audio_files)} audio files")

    # Create utterances subfolder
    utterances_dir = folder / "utterances"
    utterances_dir.mkdir(exist_ok=True)

    wav_files = []
    for i, audio_path in enumerate(audio_files, start=1):
        wav_name = f"utt{i:03d}.wav"
        wav_path = utterances_dir / wav_name

        if wav_path.exists():
            print(f"  {wav_name} already exists, skipping")
            wav_files.append(wav_path)
            continue

        print(f"  Converting {audio_path.name} -> utterances/{wav_name}")

        # Load audio (pydub auto-detects format)
        audio = AudioSegment.from_file(str(audio_path))

        # Convert to 24kHz mono
        audio = audio.set_frame_rate(24000).set_channels(1)

        # Apply subtle fade in/out (50ms each)
        audio = audio.fade_in(50).fade_out(50)

        audio.export(str(wav_path), format="wav")
        wav_files.append(wav_path)

    print(f"  Converted {len(wav_files)} files to utterances/")
    return wav_files


# ============================================================================
# Step 2: Transcribe with Whisper
# ============================================================================

def transcribe_with_whisper(wav_files: list[Path], folder: Path) -> list[dict]:
    """Transcribe all WAV files using Whisper large-v3."""
    print_step(2, "Transcribing with Whisper")

    import whisper

    # Create data subfolder
    data_dir = folder / "data"
    data_dir.mkdir(exist_ok=True)

    print("  Loading Whisper model (large-v3)...")
    model = whisper.load_model("large-v3")

    transcriptions = []
    transcript_lines = []

    for i, wav_path in enumerate(wav_files, start=1):
        print(f"  Transcribing {i}/{len(wav_files)}: {wav_path.name}")

        result = model.transcribe(
            str(wav_path),
            language="fr",
            task="transcribe"
        )

        text = result["text"].strip()
        if text:
            transcriptions.append({
                "audio": str(wav_path),
                "text": text
            })
            transcript_lines.append(f"{wav_path.name}: {text}")

    # Save human-readable transcripts
    transcripts_path = data_dir / "transcripts.txt"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))
    print(f"  Saved transcripts to data/transcripts.txt")

    # Save train_raw.jsonl
    train_raw_path = data_dir / "train_raw.jsonl"
    with open(train_raw_path, "w", encoding="utf-8") as f:
        for item in transcriptions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved raw data to data/train_raw.jsonl")

    print(f"  Transcribed {len(transcriptions)} utterances")
    return transcriptions


# ============================================================================
# Step 3: Select Reference Audio
# ============================================================================

def select_reference_audio(wav_files: list[Path], folder: Path) -> Path:
    """Select best reference audio (closest to 7 seconds, in 5-10s range)."""
    print_step(3, "Selecting Reference Audio")

    import librosa

    # Create data subfolder if needed
    data_dir = folder / "data"
    data_dir.mkdir(exist_ok=True)

    ref_path = data_dir / "ref.wav"
    if ref_path.exists():
        print(f"  Using existing data/ref.wav")
        return ref_path

    # Find best candidate
    target_duration = 7.0
    min_duration = 5.0
    max_duration = 10.0

    best_file = None
    best_diff = float('inf')

    for wav_path in wav_files:
        duration = librosa.get_duration(path=str(wav_path))

        if min_duration <= duration <= max_duration:
            diff = abs(duration - target_duration)
            if diff < best_diff:
                best_diff = diff
                best_file = wav_path
                best_duration = duration

    # Fallback: use any file if none in range
    if best_file is None:
        print("  No files in 5-10s range, using first file")
        best_file = wav_files[0]
        best_duration = librosa.get_duration(path=str(best_file))

    print(f"  Selected: {best_file.name} ({best_duration:.1f}s)")

    # Copy to ref.wav
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(str(best_file))
    audio.export(str(ref_path), format="wav")

    print(f"  Created data/ref.wav")
    return ref_path


# ============================================================================
# Step 4: Generate Audio Codes
# ============================================================================

def generate_audio_codes(transcriptions: list[dict], ref_path: Path, folder: Path) -> list[dict]:
    """Generate audio codes using Qwen3-TTS Tokenizer."""
    print_step(4, "Generating Audio Codes")

    from qwen_tts import Qwen3TTSTokenizer

    # Create data subfolder if needed
    data_dir = folder / "data"
    data_dir.mkdir(exist_ok=True)

    print("  Loading Qwen3-TTS Tokenizer (12Hz)...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0"
    )

    training_data = []
    batch_size = 32

    for batch_start in range(0, len(transcriptions), batch_size):
        batch_end = min(batch_start + batch_size, len(transcriptions))
        batch = transcriptions[batch_start:batch_end]

        print(f"  Encoding batch {batch_start+1}-{batch_end}/{len(transcriptions)}...")

        batch_audios = [item["audio"] for item in batch]

        try:
            enc_result = tokenizer.encode(batch_audios)

            for codes, item in zip(enc_result.audio_codes, batch):
                codes_list = codes.cpu().tolist()

                training_data.append({
                    "audio": item["audio"],
                    "text": item["text"],
                    "ref_audio": str(ref_path),
                    "audio_codes": codes_list
                })
        except Exception as e:
            print(f"  Warning: Batch encoding failed: {e}")
            # Fallback: encode one by one
            for item in batch:
                try:
                    enc_result = tokenizer.encode(item["audio"])
                    codes_list = enc_result.audio_codes[0].cpu().tolist()

                    training_data.append({
                        "audio": item["audio"],
                        "text": item["text"],
                        "ref_audio": str(ref_path),
                        "audio_codes": codes_list
                    })
                except Exception as e2:
                    print(f"  Warning: Failed to encode {item['audio']}: {e2}")

    # Save train_with_codes.jsonl
    output_path = data_dir / "train_with_codes.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Generated codes for {len(training_data)} utterances")
    print(f"  Saved to data/train_with_codes.jsonl")
    return training_data


# ============================================================================
# Step 5: Train Model
# ============================================================================

def train_model(
    folder: Path,
    speaker_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    gradient_accumulation_steps: int
):
    """Run the training script."""
    print_step(5, "Training Model")

    data_dir = folder / "data"
    train_jsonl = data_dir / "train_with_codes.jsonl"
    ref_audio = data_dir / "ref.wav"
    output_dir = folder / "model"

    training_script = QWEN_REPO / "finetuning" / "sft_12hz_fixed.py"

    cmd = [
        sys.executable,
        str(training_script),
        "--train_jsonl", str(train_jsonl),
        "--output_model_path", str(output_dir),
        "--ref_audio", str(ref_audio),
        "--speaker_name", speaker_name,
        "--num_epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
    ]

    print(f"  Training JSONL: {train_jsonl}")
    print(f"  Reference audio: {ref_audio}")
    print(f"  Output directory: {output_dir}")
    print(f"  Speaker name: {speaker_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print()

    # Run training
    subprocess.run(cmd, check=True)

    # Create easy-access checkpoint link/copy
    final_checkpoint = output_dir / f"checkpoint-epoch-{epochs}"
    easy_checkpoint = output_dir / "checkpoint"

    if final_checkpoint.exists():
        # Remove old easy checkpoint if exists
        if easy_checkpoint.exists():
            import shutil
            shutil.rmtree(str(easy_checkpoint))

        # Try symlink first (faster, less disk space)
        try:
            easy_checkpoint.symlink_to(final_checkpoint.name)
            print(f"\n  Created symlink: model/checkpoint -> checkpoint-epoch-{epochs}")
        except OSError:
            # Symlink failed (Windows without admin), copy instead
            import shutil
            shutil.copytree(str(final_checkpoint), str(easy_checkpoint))
            print(f"\n  Created copy: model/checkpoint (from checkpoint-epoch-{epochs})")


# ============================================================================
# Step 6: Generate Examples
# ============================================================================

# Test phrases with emotions for generating examples
EXAMPLE_PHRASES = [
    ("nat1_hesitant.wav", "Hmm... ouais, je sais pas trop...", "hesitant, thinking"),
    ("nat2_exasperated.wav", "Pfff, mais attends attends...", "exasperated"),
    ("nat3_realization.wav", "Ah! Mais oui! Du coup...", "sudden realization"),
    ("nat4_awkward.wav", "Euh... bon bah... comment dire...", "awkward, searching for words"),
    ("nat5_excited.wav", "Oh là là, c'est génial! Genre...", "enthusiastic, excited"),
    ("nat6_disapproval.wav", "Tss tss tss... non non non...", "disapproving"),
    ("nat7_honest.wav", "Bah écoute, honnêtement? Hmm...", "frank, uncertain"),
    ("nat8_laughing.wav", "Ha ha! Non mais sérieux...", "laughing, amused"),
    ("nat9_sad.wav", "Oh... euh... je... enfin...", "sad, emotional"),
    ("nat10_urgent.wav", "Woh woh woh! Attends attends!", "urgent, alarmed"),
    ("nat11_casual.wav", "Bon, alors, du coup... voilà.", "casual, wrapping up"),
    ("nat12_agreeing.wav", "Mmh mmh, oui oui, je vois...", "nodding, agreeing"),
]


def generate_examples(folder: Path, speaker_name: str, epochs: int):
    """Generate example audio files using the trained model."""
    print_step(6, "Generating Example Audio")

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # Find the checkpoint (prefer easy-access path)
    model_dir = folder / "model" / "checkpoint"
    if not model_dir.exists():
        model_dir = folder / "model" / f"checkpoint-epoch-{epochs}"
    if not model_dir.exists():
        # Try to find any checkpoint
        checkpoints = sorted((folder / "model").glob("checkpoint-epoch-*"))
        if checkpoints:
            model_dir = checkpoints[-1]
        else:
            print("  Error: No checkpoint found!")
            return

    print(f"  Loading model from: {model_dir}")

    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )

    examples_dir = folder / "examples"
    examples_dir.mkdir(exist_ok=True)

    print(f"  Generating {len(EXAMPLE_PHRASES)} examples...")

    for filename, text, instruct in EXAMPLE_PHRASES:
        output_path = examples_dir / filename
        print(f"  -> {filename}: \"{text}\" [{instruct}]")

        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker_name,
                instruct=instruct,
                language="French",
                max_new_tokens=2048,
            )

            # Save as WAV (wavs is a list, take first element)
            sf.write(str(output_path), wavs[0], sr)
            duration = len(wavs[0]) / sr
            print(f"     Saved ({duration:.1f}s)")
        except Exception as e:
            print(f"     Error: {e}")

    print(f"\n  Examples saved to {examples_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-folder voice training pipeline for Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    1. Create folder:  mkdir thalya
    2. Add .ogg files: cp *.ogg thalya/
    3. Run training:   python train_voice.py thalya
    4. Done! Model saved in thalya/model/
        """
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing .ogg files (folder name becomes speaker name)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Learning rate (default: 2e-6)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=2,
        help="Batch size per device (default: 2)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", "-g",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip OGG to WAV conversion"
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip Whisper transcription"
    )
    parser.add_argument(
        "--skip-codes",
        action="store_true",
        help="Skip audio code generation"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (just prepare data)"
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Skip generating example audio files"
    )
    parser.add_argument(
        "--only-examples",
        action="store_true",
        help="Only generate examples (skip all other steps)"
    )
    args = parser.parse_args()

    # Parse folder path
    folder = Path(args.folder).resolve()
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        sys.exit(1)

    speaker_name = folder.name

    print("=" * 60)
    print("Qwen3-TTS Single-Folder Voice Training Pipeline")
    print("=" * 60)
    print(f"\nFolder: {folder}")
    print(f"Speaker name: {speaker_name}")

    # Handle --only-examples: skip to example generation
    if args.only_examples:
        generate_examples(folder, speaker_name, args.epochs)
        print("\n" + "=" * 60)
        print("Example Generation Complete!")
        print("=" * 60)
        return

    # Step 1: Convert audio to WAV
    if args.skip_convert:
        print_step(1, "Converting Audio to WAV (SKIPPED)")
        utterances_dir = folder / "utterances"
        wav_files = sorted(utterances_dir.glob("utt*.wav"))
        if not wav_files:
            print(f"  Error: No utt*.wav files found in utterances/")
            sys.exit(1)
        print(f"  Found {len(wav_files)} existing WAV files in utterances/")
    else:
        wav_files = convert_audio_to_wav(folder)
        if not wav_files:
            print("\nError: No audio files to process")
            sys.exit(1)

    # Step 2: Transcribe
    if args.skip_transcribe:
        print_step(2, "Transcribing with Whisper (SKIPPED)")
        data_dir = folder / "data"
        train_raw_path = data_dir / "train_raw.jsonl"
        if not train_raw_path.exists():
            print(f"  Error: data/train_raw.jsonl not found")
            sys.exit(1)
        with open(train_raw_path, "r", encoding="utf-8") as f:
            transcriptions = [json.loads(line) for line in f]
        print(f"  Loaded {len(transcriptions)} existing transcriptions")
    else:
        transcriptions = transcribe_with_whisper(wav_files, folder)

    # Step 3: Select reference audio
    ref_path = select_reference_audio(wav_files, folder)

    # Step 4: Generate audio codes
    if args.skip_codes:
        print_step(4, "Generating Audio Codes (SKIPPED)")
        data_dir = folder / "data"
        train_with_codes_path = data_dir / "train_with_codes.jsonl"
        if not train_with_codes_path.exists():
            print(f"  Error: data/train_with_codes.jsonl not found")
            sys.exit(1)
        with open(train_with_codes_path, "r", encoding="utf-8") as f:
            training_data = [json.loads(line) for line in f]
        print(f"  Loaded {len(training_data)} existing training samples")
    else:
        training_data = generate_audio_codes(transcriptions, ref_path, folder)

    # Step 5: Train
    if args.skip_train:
        print_step(5, "Training Model (SKIPPED)")
        print("  Data preparation complete!")
    else:
        train_model(
            folder=folder,
            speaker_name=speaker_name,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

    # Step 6: Generate examples
    if args.skip_train or args.skip_examples:
        print_step(6, "Generating Example Audio (SKIPPED)")
    else:
        generate_examples(folder, speaker_name, args.epochs)

    # Done!
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nFolder structure:")
    print(f"  {folder.name}/")
    print(f"  ├── *.ogg/wav/mp3/...        # Original audio files")
    print(f"  ├── utterances/              # Converted 24kHz WAVs")
    print(f"  │   └── utt001.wav ...")
    print(f"  ├── data/                    # Training data")
    print(f"  │   ├── ref.wav")
    print(f"  │   ├── transcripts.txt")
    print(f"  │   ├── train_raw.jsonl")
    print(f"  │   └── train_with_codes.jsonl")
    print(f"  ├── model/                   # Trained checkpoints")
    print(f"  │   ├── checkpoint/          # Latest (easy access)")
    print(f"  │   └── checkpoint-epoch-N/")
    print(f"  └── examples/                # Generated examples")

    if not args.skip_train:
        print(f"\nTo use the trained voice:")
        print(f'  from qwen_tts import Qwen3TTSModel')
        print(f'  model = Qwen3TTSModel.from_pretrained("{folder}/model/checkpoint")')
        print(f'  model.generate_custom_voice(')
        print(f'      text="Bonjour, comment allez-vous?",')
        print(f'      speaker="{speaker_name}",')
        print(f'      instruct="warmly, with a smile"')
        print(f'  )')


if __name__ == "__main__":
    main()
