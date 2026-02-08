#!/usr/bin/env python3
"""
Complete pipeline for processing podcasts into TTS training data.

Pipeline steps:
1. Download podcasts from URLs (yt-dlp for YouTube, requests for direct links)
2. Convert to 24kHz mono WAV
3. Detect silence using pydub silence detection (not VAD)
4. Split at silence boundaries - minimum 800ms silence, min segment 2s, max segment 15s
5. Filter music using spectral analysis heuristics
6. Add silence padding (150ms) and fade (30ms) to each segment
7. Transcribe with Whisper (large-v3)
8. Generate train.jsonl with audio codes
9. Output ready for training

Usage:
    python process_podcasts.py [--skip-download] [--skip-segment] [--skip-transcribe]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = BASE_DIR / "training_data"
RAW_PODCASTS_DIR = TRAINING_DATA_DIR / "raw_podcasts"
SEGMENTS_DIR = TRAINING_DATA_DIR / "segments"
PODCAST_URLS_FILE = BASE_DIR / "podcast_urls.json"

# Default settings (can be overridden in podcast_urls.json)
DEFAULT_SETTINGS = {
    "min_silence_ms": 800,
    "silence_thresh_db": -40,
    "min_segment_sec": 2.0,
    "max_segment_sec": 15.0,
    "silence_padding_ms": 150,
    "fade_ms": 150,
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_config() -> dict:
    """Load podcast URLs and settings from JSON file."""
    if not PODCAST_URLS_FILE.exists():
        print(f"Error: {PODCAST_URLS_FILE} not found. Please create it with your podcast URLs.")
        sys.exit(1)

    with open(PODCAST_URLS_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Merge default settings with user settings
    settings = DEFAULT_SETTINGS.copy()
    if "settings" in config:
        settings.update(config["settings"])
    config["settings"] = settings

    return config


def ensure_dirs():
    """Create necessary directories."""
    TRAINING_DATA_DIR.mkdir(exist_ok=True)
    RAW_PODCASTS_DIR.mkdir(exist_ok=True)
    SEGMENTS_DIR.mkdir(exist_ok=True)


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use hash of URL
    return f"url_{hash(url) % 100000:05d}"


# ============================================================================
# Step 1: Download Podcasts
# ============================================================================

def download_podcast(url: str, output_dir: Path) -> Optional[Path]:
    """Download podcast audio from URL using yt-dlp."""
    video_id = extract_video_id(url)
    output_path = output_dir / f"podcast_{video_id}.wav"

    if output_path.exists():
        print(f"  Already downloaded: {output_path.name}")
        return output_path

    print(f"  Downloading: {url}")

    # Use yt-dlp to download and convert to 24kHz mono WAV
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
        "-o", str(output_dir / f"podcast_{video_id}.%(ext)s"),
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if output_path.exists():
            print(f"  Downloaded: {output_path.name}")
            return output_path
        else:
            # yt-dlp might save with different extension, convert if needed
            for ext in [".m4a", ".mp3", ".opus", ".webm"]:
                temp_path = output_dir / f"podcast_{video_id}{ext}"
                if temp_path.exists():
                    convert_to_wav(temp_path, output_path)
                    temp_path.unlink()
                    return output_path
            print(f"  Warning: Could not find downloaded file for {url}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"  Error downloading {url}: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  Error: yt-dlp not found. Install it with: pip install yt-dlp")
        return None


def convert_to_wav(input_path: Path, output_path: Path):
    """Convert audio file to 24kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "24000", "-ac", "1",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def download_all_podcasts(config: dict) -> List[Path]:
    """Download all podcasts from config."""
    print("\n=== Step 1: Downloading Podcasts ===")

    downloaded = []
    for item in config.get("podcasts", []):
        url = item.get("url", "")
        if not url or url.startswith("https://www.youtube.com/watch?v=EXAMPLE"):
            continue

        path = download_podcast(url, RAW_PODCASTS_DIR)
        if path:
            downloaded.append(path)

    # Also include any existing WAV files in raw_podcasts
    for wav_file in RAW_PODCASTS_DIR.glob("*.wav"):
        if wav_file not in downloaded:
            downloaded.append(wav_file)

    print(f"  Total podcasts available: {len(downloaded)}")
    return downloaded


# ============================================================================
# Step 2: Silence-Based Segmentation
# ============================================================================

def segment_by_silence(
    audio_path: Path,
    settings: dict
) -> List[Tuple[int, int]]:
    """
    Segment audio at silence boundaries.

    Returns list of (start_ms, end_ms) tuples.
    """
    min_silence_ms = settings["min_silence_ms"]
    silence_thresh_db = settings["silence_thresh_db"]
    min_segment_ms = int(settings["min_segment_sec"] * 1000)
    max_segment_ms = int(settings["max_segment_sec"] * 1000)

    # Load audio with pydub
    audio = AudioSegment.from_wav(str(audio_path))

    # Detect non-silent regions
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db
    )

    if not nonsilent_ranges:
        return []

    # Post-process: merge short segments, split long segments
    segments = []
    current_start = nonsilent_ranges[0][0]
    current_end = nonsilent_ranges[0][1]

    for i in range(1, len(nonsilent_ranges)):
        start, end = nonsilent_ranges[i]

        # Check if we should merge with previous segment
        gap = start - current_end
        merged_length = end - current_start

        if merged_length <= max_segment_ms and gap < min_silence_ms * 2:
            # Merge: extend current segment
            current_end = end
        else:
            # Don't merge: save current segment if long enough
            if current_end - current_start >= min_segment_ms:
                segments.append((current_start, current_end))
            current_start = start
            current_end = end

    # Don't forget the last segment
    if current_end - current_start >= min_segment_ms:
        segments.append((current_start, current_end))

    # Split segments that are too long
    final_segments = []
    for start, end in segments:
        duration = end - start
        if duration <= max_segment_ms:
            final_segments.append((start, end))
        else:
            # Split at natural silence points within segment
            sub_audio = audio[start:end]
            sub_nonsilent = detect_nonsilent(
                sub_audio,
                min_silence_len=min_silence_ms // 2,
                silence_thresh=silence_thresh_db + 5
            )

            if len(sub_nonsilent) > 1:
                # Try to split at internal silence boundaries
                sub_start = 0
                for j, (s, e) in enumerate(sub_nonsilent):
                    if e - sub_start >= max_segment_ms * 0.8:
                        final_segments.append((start + sub_start, start + s))
                        sub_start = s
                if end - (start + sub_start) >= min_segment_ms:
                    final_segments.append((start + sub_start, end))
            else:
                # Force split in the middle
                mid = start + duration // 2
                final_segments.append((start, mid))
                final_segments.append((mid, end))

    return final_segments


def is_music(audio_segment: np.ndarray, sr: int = 24000) -> bool:
    """
    Detect if audio segment is likely music using spectral analysis.

    Heuristics:
    - High spectral flatness = noise/music
    - Low zero crossing rate variance = music
    - High spectral centroid variance = speech
    """
    if len(audio_segment) < sr:  # Too short to analyze
        return False

    try:
        # Compute spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio_segment, n_fft=2048)
        mean_flatness = np.mean(flatness)

        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_segment)
        zcr_std = np.std(zcr)

        # Compute spectral centroid variance
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        centroid_std = np.std(centroid) / (np.mean(centroid) + 1e-6)

        # Music tends to have:
        # - Higher spectral flatness (> 0.1)
        # - Lower ZCR variance (< 0.05)
        # - Lower relative centroid variance (< 0.3)
        is_likely_music = (
            mean_flatness > 0.15 and
            zcr_std < 0.04 and
            centroid_std < 0.25
        )

        return is_likely_music
    except Exception:
        return False


def apply_smart_fade(segment: AudioSegment, fade_ms: int = 500, threshold: int = 500) -> AudioSegment:
    """
    Apply fade in/out at actual speech boundaries, not segment boundaries.

    This finds where speech actually starts/ends (based on amplitude threshold)
    and applies the fade there, avoiding fading silence.
    """
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    sample_rate = segment.frame_rate
    window = sample_rate // 20  # 50ms window for detection

    # Find speech start (first window with amplitude above threshold)
    speech_start = 0
    for i in range(0, len(samples) - window, window):
        if np.max(np.abs(samples[i:i+window])) > threshold:
            speech_start = max(0, i - window)  # back up a bit
            break

    # Find speech end (last window with amplitude above threshold)
    speech_end = len(samples)
    for i in range(len(samples) - window, 0, -window):
        if np.max(np.abs(samples[i:i+window])) > threshold:
            speech_end = min(len(samples), i + window * 2)  # extend a bit
            break

    # Calculate fade length in samples
    fade_samples = int(fade_ms * sample_rate / 1000)

    # Apply fade in at speech start
    fade_in_end = min(speech_start + fade_samples, speech_end)
    fade_len = fade_in_end - speech_start
    if fade_len > 0:
        fade_in_curve = np.linspace(0, 1, fade_len)
        samples[speech_start:fade_in_end] *= fade_in_curve

    # Apply fade out at speech end
    fade_out_start = max(speech_end - fade_samples, speech_start)
    fade_len = speech_end - fade_out_start
    if fade_len > 0:
        fade_out_curve = np.linspace(1, 0, fade_len)
        samples[fade_out_start:speech_end] *= fade_out_curve

    # Convert back to AudioSegment
    samples = samples.astype(np.int16)
    return AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=segment.sample_width,
        channels=segment.channels
    )


def segment_all_podcasts(podcast_files: List[Path], settings: dict) -> List[Path]:
    """Segment all podcasts and save to segments directory."""
    print("\n=== Step 2: Segmenting at Silence Boundaries ===")

    padding_ms = settings["silence_padding_ms"]
    fade_ms = settings["fade_ms"]

    segment_files = []
    segment_idx = 1

    for podcast_path in podcast_files:
        print(f"  Processing: {podcast_path.name}")

        # Load full audio for music detection
        audio_np, sr = librosa.load(str(podcast_path), sr=24000, mono=True)
        audio_pydub = AudioSegment.from_wav(str(podcast_path))

        # Get segment boundaries
        segments = segment_by_silence(podcast_path, settings)
        print(f"    Found {len(segments)} potential segments")

        music_count = 0
        for start_ms, end_ms in segments:
            # Extract segment
            segment = audio_pydub[start_ms:end_ms]

            # Convert to numpy for music detection
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            segment_np = audio_np[start_sample:end_sample]

            # Skip if likely music
            if is_music(segment_np, sr):
                music_count += 1
                continue

            # Simple fade in/out (50ms each - subtle, not noticeable)
            segment = segment.fade_in(50).fade_out(50)

            # No silence padding - causes identical start tokens in training

            # Save segment
            output_path = SEGMENTS_DIR / f"seg_{segment_idx:04d}.wav"
            segment.export(str(output_path), format="wav")
            segment_files.append(output_path)
            segment_idx += 1

        if music_count > 0:
            print(f"    Filtered {music_count} music segments")

    print(f"  Total segments: {len(segment_files)}")
    return segment_files


# ============================================================================
# Step 3: Transcribe with Whisper
# ============================================================================

def transcribe_segments(segment_files: List[Path]) -> List[dict]:
    """Transcribe segments using Whisper."""
    print("\n=== Step 3: Transcribing with Whisper ===")

    try:
        import whisper
    except ImportError:
        print("  Error: openai-whisper not installed. Install with: pip install openai-whisper")
        sys.exit(1)

    print("  Loading Whisper model (large-v3)...")
    model = whisper.load_model("large-v3")

    transcriptions = []
    for i, segment_path in enumerate(segment_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Transcribing {i+1}/{len(segment_files)}...")

        result = model.transcribe(
            str(segment_path),
            language="fr",  # Adjust as needed
            task="transcribe"
        )

        text = result["text"].strip()
        if text:
            transcriptions.append({
                "audio": str(segment_path),
                "text": text
            })

    print(f"  Transcribed {len(transcriptions)} segments")
    return transcriptions


# ============================================================================
# Step 4: Generate Audio Codes
# ============================================================================

def generate_audio_codes(transcriptions: List[dict], ref_audio_path: Path) -> List[dict]:
    """Generate audio codes for training using Qwen3-TTS Tokenizer."""
    print("\n=== Step 4: Generating Audio Codes ===")

    # Add qwen3-tts-repo to path
    qwen_repo = BASE_DIR / "qwen3-tts-repo"
    if str(qwen_repo) not in sys.path:
        sys.path.insert(0, str(qwen_repo))

    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as e:
        print(f"  Error importing Qwen3TTSTokenizer: {e}")
        print("  Make sure qwen3-tts-repo is set up correctly")
        sys.exit(1)

    print("  Loading Qwen3-TTS Tokenizer (12Hz)...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0"
    )

    # Process in batches for efficiency
    BATCH_SIZE = 32
    training_data = []

    for batch_start in range(0, len(transcriptions), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(transcriptions))
        batch = transcriptions[batch_start:batch_end]

        print(f"  Encoding batch {batch_start+1}-{batch_end}/{len(transcriptions)}...")

        # Collect audio paths for this batch
        batch_audios = [item["audio"] for item in batch]

        try:
            # Encode batch
            enc_result = tokenizer.encode(batch_audios)

            # Process each result
            for i, (codes, item) in enumerate(zip(enc_result.audio_codes, batch)):
                audio_path = item["audio"]
                try:
                    # Convert to list format expected by dataset
                    codes_list = codes.cpu().tolist()  # [time, 16]

                    training_data.append({
                        "audio": str(Path(audio_path).relative_to(BASE_DIR)),
                        "text": item["text"],
                        "ref_audio": str(ref_audio_path.relative_to(BASE_DIR)),
                        "audio_codes": codes_list
                    })
                except Exception as e:
                    print(f"  Warning: Failed to process codes for {audio_path}: {e}")

        except Exception as e:
            print(f"  Warning: Batch encoding failed: {e}")
            # Try encoding one by one as fallback
            for item in batch:
                audio_path = item["audio"]
                try:
                    enc_result = tokenizer.encode(audio_path)
                    codes_list = enc_result.audio_codes[0].cpu().tolist()

                    training_data.append({
                        "audio": str(Path(audio_path).relative_to(BASE_DIR)),
                        "text": item["text"],
                        "ref_audio": str(ref_audio_path.relative_to(BASE_DIR)),
                        "audio_codes": codes_list
                    })
                except Exception as e2:
                    print(f"  Warning: Failed to encode {audio_path}: {e2}")

    print(f"  Generated codes for {len(training_data)} utterances")
    return training_data


# ============================================================================
# Step 5: Copy Segments to Final Location
# ============================================================================

def copy_segments_to_training(segment_files: List[Path]) -> dict:
    """Copy segments to training_data as uttXXX.wav files."""
    print("\n=== Step 5: Organizing Training Files ===")

    mapping = {}
    for i, seg_path in enumerate(segment_files, start=1):
        utt_name = f"utt{i:03d}.wav"
        dest_path = TRAINING_DATA_DIR / utt_name

        # Copy file
        audio = AudioSegment.from_wav(str(seg_path))
        audio.export(str(dest_path), format="wav")

        mapping[str(seg_path)] = str(dest_path)

    print(f"  Copied {len(mapping)} files")
    return mapping


# ============================================================================
# Step 6: Create Reference Audio
# ============================================================================

def create_ref_audio():
    """Create reference audio from maryam.ogg if ref.wav doesn't exist."""
    ref_wav = TRAINING_DATA_DIR / "ref.wav"
    maryam_ogg = TRAINING_DATA_DIR / "maryam.ogg"

    if ref_wav.exists():
        print(f"  Reference audio exists: {ref_wav}")
        return ref_wav

    if maryam_ogg.exists():
        print(f"  Converting maryam.ogg to ref.wav...")
        audio = AudioSegment.from_ogg(str(maryam_ogg))
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(str(ref_wav), format="wav")
        return ref_wav

    print("  Warning: No reference audio found (maryam.ogg or ref.wav)")
    return None


# ============================================================================
# Step 7: Write Training JSONL
# ============================================================================

def write_training_jsonl(training_data: List[dict], path_mapping: dict):
    """Write final training JSONL file."""
    print("\n=== Step 6: Writing Training JSONL ===")

    # Update paths in training data
    for item in training_data:
        old_audio = item["audio"]
        if old_audio in path_mapping:
            item["audio"] = path_mapping[old_audio]

    output_path = TRAINING_DATA_DIR / "train_with_codes.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Written {len(training_data)} entries to {output_path}")
    return output_path


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process podcasts for TTS training")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-segment", action="store_true", help="Skip segmentation step")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription step")
    parser.add_argument("--skip-codes", action="store_true", help="Skip audio code generation")
    args = parser.parse_args()

    # Setup
    ensure_dirs()
    config = load_config()
    settings = config["settings"]

    print("=" * 60)
    print("Qwen3-TTS Voice Cloning Pipeline")
    print("=" * 60)
    print(f"Settings: {json.dumps(settings, indent=2)}")

    # Create reference audio
    ref_audio = create_ref_audio()
    if not ref_audio:
        print("Error: Reference audio is required. Add maryam.ogg to training_data/")
        sys.exit(1)

    # Step 1: Download
    if args.skip_download:
        print("\n=== Step 1: Download (SKIPPED) ===")
        podcast_files = list(RAW_PODCASTS_DIR.glob("*.wav"))
    else:
        podcast_files = download_all_podcasts(config)

    if not podcast_files:
        print("No podcast files found. Add URLs to podcast_urls.json")
        sys.exit(1)

    # Step 2: Segment
    if args.skip_segment:
        print("\n=== Step 2: Segmentation (SKIPPED) ===")
        segment_files = sorted(SEGMENTS_DIR.glob("*.wav"))
    else:
        segment_files = segment_all_podcasts(podcast_files, settings)

    if not segment_files:
        print("No segments created. Check your podcast files.")
        sys.exit(1)

    # Step 3: Transcribe
    if args.skip_transcribe:
        print("\n=== Step 3: Transcription (SKIPPED) ===")
        # Try to load existing transcriptions
        train_raw_path = BASE_DIR / "train_raw.jsonl"
        if train_raw_path.exists():
            with open(train_raw_path, "r", encoding="utf-8") as f:
                transcriptions = [json.loads(line) for line in f]
        else:
            print("Error: No existing transcriptions found")
            sys.exit(1)
    else:
        transcriptions = transcribe_segments(segment_files)

        # Save raw transcriptions
        train_raw_path = BASE_DIR / "train_raw.jsonl"
        with open(train_raw_path, "w", encoding="utf-8") as f:
            for item in transcriptions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Step 4: Copy segments to training folder
    path_mapping = copy_segments_to_training(segment_files)

    # Update transcription paths
    for item in transcriptions:
        old_path = item["audio"]
        if old_path in path_mapping:
            item["audio"] = path_mapping[old_path]

    # Step 5: Generate audio codes
    if args.skip_codes:
        print("\n=== Step 4: Audio Codes (SKIPPED) ===")
        # Load existing codes
        existing_path = TRAINING_DATA_DIR / "train_with_codes.jsonl"
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                training_data = [json.loads(line) for line in f]
        else:
            print("Error: No existing audio codes found")
            sys.exit(1)
    else:
        training_data = generate_audio_codes(transcriptions, ref_audio)

    # Step 6: Write final JSONL
    write_training_jsonl(training_data, {})

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review segments in {SEGMENTS_DIR}")
    print(f"  2. Run training with:")
    print(f"     python qwen3-tts-repo/finetuning/sft_12hz_fixed.py \\")
    print(f"       --train_jsonl training_data/train_with_codes.jsonl \\")
    print(f"       --output_model_path output_final \\")
    print(f"       --num_epochs 30 \\")
    print(f"       --lr 5e-5 \\")
    print(f"       --batch_size 4 \\")
    print(f"       --speaker_name maryam")


if __name__ == "__main__":
    main()
