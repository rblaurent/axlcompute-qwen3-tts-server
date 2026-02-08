"""
Improved podcast processor with:
- Voice Activity Detection (VAD) to isolate clean speech
- Music/jingle detection and removal
- Audio quality filtering
- Better segmentation
"""

import whisper
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from silero_vad import load_silero_vad, get_speech_timestamps

# Configuration
INPUT_DIR = Path("training_data/raw_podcasts")
OUTPUT_DIR = Path("training_data")
TARGET_SR = 24000
MIN_DURATION = 3.0
MAX_DURATION = 12.0  # Shorter max for better quality
MIN_SPEECH_RATIO = 0.85  # At least 85% speech in segment
MIN_ENERGY_DB = -35  # Minimum audio energy (filters silence)
MAX_MUSIC_SCORE = 0.95  # Disabled - VAD handles non-speech filtering


def load_vad_model():
    """Load Silero VAD model."""
    print("Loading VAD model...")
    model = load_silero_vad()
    return model


def detect_speech_timestamps(audio: np.ndarray, sr: int, vad_model) -> list:
    """Get speech timestamps using Silero VAD."""
    # VAD needs 16kHz
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    audio_tensor = torch.from_numpy(audio_16k).float()

    timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=16000,
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )

    # Convert to original sample rate
    scale = sr / 16000
    return [{"start": int(ts["start"] * scale), "end": int(ts["end"] * scale)} for ts in timestamps]


def compute_music_score(audio: np.ndarray, sr: int) -> float:
    """
    Estimate how "musical" a segment is.
    Music typically has:
    - More harmonic content
    - Steady rhythm
    - Less dynamic range in pitch

    Returns score 0-1 (higher = more likely music)
    """
    try:
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

        # Music tends to have more stable spectral features
        centroid_std = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-6)

        # Compute harmonic ratio
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(audio)) + 1e-6)

        # Music has higher harmonic ratio and more stable spectrum
        music_score = (harmonic_ratio * 0.5) + ((1 - min(centroid_std, 1)) * 0.5)

        # Check for steady beat (music indicator)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        if tempo > 60 and tempo < 180:  # Typical music tempo range
            music_score += 0.2

        return min(music_score, 1.0)
    except:
        return 0.0


def compute_speech_quality(audio: np.ndarray, sr: int) -> dict:
    """Compute speech quality metrics."""
    # RMS energy in dB
    rms = np.sqrt(np.mean(audio**2))
    energy_db = 20 * np.log10(rms + 1e-10)

    # Zero crossing rate (speech has moderate ZCR)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

    # Spectral flatness (noise has high flatness, speech is lower)
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])

    return {
        "energy_db": energy_db,
        "zcr": zcr,
        "flatness": flatness,
        "is_good": energy_db > MIN_ENERGY_DB and flatness < 0.5
    }


def process_podcast(audio_path: Path, vad_model, whisper_model) -> list:
    """Process a single podcast file."""
    print(f"\nProcessing: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Step 1: Get speech timestamps using VAD
    print("  Running VAD...")
    speech_timestamps = detect_speech_timestamps(audio, sr, vad_model)
    print(f"  Found {len(speech_timestamps)} speech regions")

    # Step 2: Filter and merge speech regions
    good_segments = []

    for ts in speech_timestamps:
        start_sample = ts["start"]
        end_sample = ts["end"]
        segment_audio = audio[start_sample:end_sample]
        segment_duration = len(segment_audio) / sr

        # Skip if too short or too long
        if segment_duration < MIN_DURATION or segment_duration > MAX_DURATION * 1.5:
            continue

        # Check music score
        music_score = compute_music_score(segment_audio, sr)
        if music_score > MAX_MUSIC_SCORE:
            print(f"    Skipping segment (music detected: {music_score:.2f})")
            continue

        # Check speech quality
        quality = compute_speech_quality(segment_audio, sr)
        if not quality["is_good"]:
            print(f"    Skipping segment (low quality: {quality['energy_db']:.1f}dB)")
            continue

        good_segments.append({
            "start": start_sample,
            "end": end_sample,
            "duration": segment_duration,
            "music_score": music_score,
            "energy_db": quality["energy_db"]
        })

    print(f"  {len(good_segments)} segments passed quality filter")

    # Step 3: Transcribe good segments
    print("  Transcribing...")
    results = []

    for i, seg in enumerate(good_segments):
        segment_audio = audio[seg["start"]:seg["end"]]

        # Transcribe this segment
        result = whisper_model.transcribe(
            segment_audio,
            language="fr",
            fp16=False,
        )

        text = result["text"].strip()
        if len(text) < 10:  # Skip very short transcriptions
            continue

        # Further split if needed based on Whisper segments
        for wseg in result["segments"]:
            wseg_text = wseg["text"].strip()
            wseg_start = seg["start"] + int(wseg["start"] * sr)
            wseg_end = seg["start"] + int(wseg["end"] * sr)
            wseg_duration = (wseg_end - wseg_start) / sr

            if wseg_duration >= MIN_DURATION and wseg_duration <= MAX_DURATION and len(wseg_text) >= 10:
                results.append({
                    "start": wseg_start,
                    "end": wseg_end,
                    "duration": wseg_duration,
                    "text": wseg_text,
                    "source": audio_path.name
                })

    print(f"  Generated {len(results)} training segments")
    return results, audio, sr


def main():
    print("=" * 60)
    print("Improved Podcast Processor")
    print("With VAD, music detection, and quality filtering")
    print("=" * 60)

    # Create directories
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check for input files
    audio_files = list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.mp3"))

    if not audio_files:
        print(f"\nNo audio files found in {INPUT_DIR}")
        print("Please place your podcast audio files there.")
        print("\nTo download from YouTube:")
        print("  yt-dlp -x --audio-format wav -o 'training_data/raw_podcasts/%(title)s.%(ext)s' URL")
        return

    print(f"\nFound {len(audio_files)} audio files")

    # Load models
    vad_model = load_vad_model()
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("medium", device="cuda")

    # Process all podcasts
    all_segments = []
    all_audio_data = {}

    for audio_path in audio_files:
        segments, audio, sr = process_podcast(audio_path, vad_model, whisper_model)
        all_segments.extend(segments)
        all_audio_data[audio_path.name] = (audio, sr)

    print(f"\n{'='*60}")
    print(f"Total segments: {len(all_segments)}")

    if not all_segments:
        print("No segments found! Check your audio files.")
        return

    # Export segments
    print("\nExporting segments...")

    # Clear old utterance files
    for old_file in OUTPUT_DIR.glob("utt*.wav"):
        old_file.unlink()

    exported = []
    for i, seg in enumerate(all_segments):
        audio, sr = all_audio_data[seg["source"]]
        segment_audio = audio[seg["start"]:seg["end"]]

        filename = f"utt{i+1:03d}.wav"
        filepath = OUTPUT_DIR / filename
        sf.write(str(filepath), segment_audio, sr)

        exported.append({
            "filename": filename,
            "text": seg["text"],
            "duration": seg["duration"]
        })

        if i < 5 or i % 50 == 0:
            print(f"  {filename}: {seg['duration']:.1f}s - {seg['text'][:40]}...")

    # Select best reference audio (longest clean segment)
    best_ref = max(all_segments, key=lambda x: x["duration"] if x["duration"] <= 10 else 0)
    audio, sr = all_audio_data[best_ref["source"]]
    ref_audio = audio[best_ref["start"]:best_ref["end"]]
    sf.write(str(OUTPUT_DIR / "ref.wav"), ref_audio, sr)

    # Write transcripts
    with open("transcripts.txt", "w", encoding="utf-8") as f:
        f.write(f"ref.wav|{best_ref['text']}\n")
        for item in exported:
            f.write(f"{item['filename']}|{item['text']}\n")

    total_duration = sum(s["duration"] for s in all_segments)
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total clean segments: {len(exported)}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Reference audio: ref.wav ({best_ref['duration']:.1f}s)")
    print(f"\nNext: Run setup_finetuning.py then fine-tune with more epochs")


if __name__ == "__main__":
    main()
