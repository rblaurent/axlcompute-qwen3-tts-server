"""
Process podcast audio for Qwen3-TTS fine-tuning:
1. Transcribe with Whisper
2. Segment into 3-15 second clips based on sentence boundaries
3. Generate training files and transcripts
"""

import whisper
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json

# Configuration
INPUT_AUDIO = Path("training_data/podcast_raw.wav")
OUTPUT_DIR = Path("training_data")
TARGET_SR = 24000
MIN_DURATION = 3.0   # Minimum clip duration in seconds
MAX_DURATION = 15.0  # Maximum clip duration in seconds

def load_and_resample(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio and resample to target sample rate."""
    print(f"Loading {audio_path}...")
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    return audio, TARGET_SR

def transcribe_audio(audio_path: Path, model_name: str = "medium") -> dict:
    """Transcribe audio using Whisper with word timestamps."""
    print(f"Loading Whisper {model_name} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    print("Transcribing (this may take a few minutes)...")
    result = model.transcribe(
        str(audio_path),
        language="fr",  # French
        word_timestamps=True,
        verbose=False
    )
    return result

def segment_by_sentences(transcription: dict, min_dur: float, max_dur: float) -> list[dict]:
    """
    Create segments based on Whisper's sentence-level segments,
    respecting min/max duration constraints.
    """
    segments = []

    for seg in transcription["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        duration = end - start

        # Skip very short or empty segments
        if duration < min_dur or not text:
            continue

        # If segment is within limits, use it directly
        if duration <= max_dur:
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "duration": duration
            })
        else:
            # Long segment - try to split by words if available
            if "words" in seg and seg["words"]:
                words = seg["words"]
                current_start = words[0]["start"]
                current_words = []

                for word in words:
                    current_words.append(word["word"])
                    current_duration = word["end"] - current_start

                    # Check if we should end this sub-segment
                    # End at punctuation or when approaching max duration
                    is_sentence_end = any(p in word["word"] for p in ".!?")

                    if (current_duration >= min_dur and is_sentence_end) or current_duration >= max_dur * 0.9:
                        sub_text = "".join(current_words).strip()
                        if sub_text:
                            segments.append({
                                "start": current_start,
                                "end": word["end"],
                                "text": sub_text,
                                "duration": word["end"] - current_start
                            })
                        current_start = word["end"]
                        current_words = []

                # Don't forget remaining words
                if current_words:
                    sub_text = "".join(current_words).strip()
                    if sub_text and (words[-1]["end"] - current_start) >= min_dur:
                        segments.append({
                            "start": current_start,
                            "end": words[-1]["end"],
                            "text": sub_text,
                            "duration": words[-1]["end"] - current_start
                        })

    return segments

def export_segments(audio: np.ndarray, sr: int, segments: list[dict], output_dir: Path) -> list[dict]:
    """Export audio segments as individual WAV files."""
    output_dir.mkdir(exist_ok=True)
    exported = []

    for i, seg in enumerate(segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        # Extract segment
        segment_audio = audio[start_sample:end_sample]

        # Save
        filename = f"utt{i+1:03d}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), segment_audio, sr)

        exported.append({
            "filename": filename,
            "text": seg["text"],
            "duration": seg["duration"]
        })

        print(f"  {filename}: {seg['duration']:.1f}s - {seg['text'][:50]}...")

    return exported

def create_reference_audio(audio: np.ndarray, sr: int, segments: list[dict], output_dir: Path):
    """Create reference audio from the clearest/longest segment."""
    # Find a good reference segment (5-10 seconds, clean speech)
    candidates = [s for s in segments if 5.0 <= s["duration"] <= 10.0]
    if not candidates:
        candidates = sorted(segments, key=lambda x: x["duration"], reverse=True)[:5]

    # Use the first good candidate
    ref_seg = candidates[0]
    start_sample = int(ref_seg["start"] * sr)
    end_sample = int(ref_seg["end"] * sr)

    ref_audio = audio[start_sample:end_sample]
    ref_path = output_dir / "ref.wav"
    sf.write(str(ref_path), ref_audio, sr)

    print(f"\nReference audio: ref.wav ({ref_seg['duration']:.1f}s)")
    print(f"  Text: {ref_seg['text']}")

    return {
        "filename": "ref.wav",
        "text": ref_seg["text"],
        "duration": ref_seg["duration"]
    }

def write_transcripts(exported: list[dict], ref: dict, output_path: Path):
    """Write transcripts.txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated transcripts from podcast\n")
        f.write("# Review and correct any transcription errors before fine-tuning\n\n")
        f.write(f"ref.wav|{ref['text']}\n\n")

        for item in exported:
            if item["filename"] != "ref.wav":
                f.write(f"{item['filename']}|{item['text']}\n")

    print(f"\nTranscripts written to: {output_path}")

def main():
    print("=" * 60)
    print("Podcast Processing for Qwen3-TTS Fine-Tuning")
    print("=" * 60)

    # Load and resample audio
    audio, sr = load_and_resample(INPUT_AUDIO)
    print(f"Audio loaded: {len(audio)/sr:.1f}s at {sr}Hz")

    # Transcribe
    transcription = transcribe_audio(INPUT_AUDIO, model_name="medium")

    # Save full transcription for reference
    with open(OUTPUT_DIR / "full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)
    print(f"Full transcription saved to: {OUTPUT_DIR / 'full_transcription.json'}")

    # Segment
    print(f"\nSegmenting into {MIN_DURATION}-{MAX_DURATION}s clips...")
    segments = segment_by_sentences(transcription, MIN_DURATION, MAX_DURATION)
    print(f"Created {len(segments)} segments")

    # Export segments
    print("\nExporting segments:")
    exported = export_segments(audio, sr, segments, OUTPUT_DIR)

    # Create reference audio
    ref = create_reference_audio(audio, sr, segments, OUTPUT_DIR)

    # Write transcripts
    write_transcripts(exported, ref, Path("transcripts.txt"))

    # Summary
    total_duration = sum(s["duration"] for s in exported)
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total segments: {len(exported)}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"\nNext steps:")
    print("  1. Review transcripts.txt for errors")
    print("  2. Run: python setup_finetuning.py")
    print("  3. Run fine-tuning")

if __name__ == "__main__":
    main()
