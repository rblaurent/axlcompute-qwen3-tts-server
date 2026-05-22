"""Prepare Springs training folder from extracted Borderlands TPS French voice lines."""

import json
from pathlib import Path
from pydub import AudioSegment

SOURCE_DIR = Path("bl_tps_french_voices/voices/Springs")
OUTPUT_DIR = Path("springs")
SAMPLE_RATE = 24000
FADE_MS = 50
REF_TARGET = 7.0
REF_MIN = 5.0
REF_MAX = 10.0


def main():
    utt_dir = OUTPUT_DIR / "utterances"
    data_dir = OUTPUT_DIR / "data"
    utt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Read transcripts (format: filename|text)
    entries = []
    for line in (SOURCE_DIR / "transcript.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        filename, text = line.split("|", 1)
        entries.append((filename.strip(), text.strip()))

    print(f"Found {len(entries)} transcript entries")

    transcript_lines = []
    jsonl_lines = []
    best_ref = None  # (utt_path, abs_diff)

    for i, (filename, text) in enumerate(entries, 1):
        utt_name = f"utt{i:03d}.wav"
        src_path = SOURCE_DIR / filename
        dst_path = utt_dir / utt_name

        # Load, resample to 24kHz mono, apply fades
        audio = AudioSegment.from_file(str(src_path))
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio = audio.fade_in(FADE_MS).fade_out(FADE_MS)
        audio.export(str(dst_path), format="wav")

        # Track ref audio candidate
        duration = len(audio) / 1000.0
        if REF_MIN <= duration <= REF_MAX:
            diff = abs(duration - REF_TARGET)
            if best_ref is None or diff < best_ref[1]:
                best_ref = (dst_path, diff, duration)

        transcript_lines.append(f"{utt_name}: {text}")
        jsonl_lines.append(json.dumps({
            "audio": str(dst_path.resolve()),
            "text": text,
        }, ensure_ascii=False))

        if i % 50 == 0:
            print(f"  Processed {i}/{len(entries)}")

    # Write transcripts.txt
    (data_dir / "transcripts.txt").write_text("\n".join(transcript_lines), encoding="utf-8")

    # Write train_raw.jsonl
    (data_dir / "train_raw.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

    # Copy ref audio
    if best_ref:
        ref_dst = data_dir / "ref.wav"
        audio = AudioSegment.from_file(str(best_ref[0]))
        audio.export(str(ref_dst), format="wav")
        print(f"Reference audio: {best_ref[0].name} ({best_ref[2]:.1f}s)")
    else:
        print("WARNING: No audio in 5-10s range found for ref.wav!")

    print(f"Done! {len(entries)} utterances in {utt_dir}")
    print(f"  transcripts.txt: {len(transcript_lines)} lines")
    print(f"  train_raw.jsonl: {len(jsonl_lines)} lines")


if __name__ == "__main__":
    main()
