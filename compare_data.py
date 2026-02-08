"""Compare old vs new training data - check for silence patterns."""
import json
import numpy as np

def analyze_codes(path, name):
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)

    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Collect all first codes (column 0) from all samples
    all_first_codes = []
    all_last_codes = []
    first_frame_codes = []  # First frame of each sample
    last_frame_codes = []   # Last frame of each sample

    for d in data:
        codes = d.get('audio_codes', [])
        if codes and len(codes) > 0:
            # First frame (start of audio)
            first_frame_codes.append(codes[0][0])  # First codec layer 0
            # Last frame (end of audio)
            last_frame_codes.append(codes[-1][0])

    print(f"\nFirst frame codec[0] values (start of audio):")
    print(f"  Unique values: {len(set(first_frame_codes))}")
    print(f"  Most common: {max(set(first_frame_codes), key=first_frame_codes.count)} (appears {first_frame_codes.count(max(set(first_frame_codes), key=first_frame_codes.count))} times)")
    print(f"  Sample: {first_frame_codes[:20]}")

    print(f"\nLast frame codec[0] values (end of audio):")
    print(f"  Unique values: {len(set(last_frame_codes))}")
    print(f"  Most common: {max(set(last_frame_codes), key=last_frame_codes.count)} (appears {last_frame_codes.count(max(set(last_frame_codes), key=last_frame_codes.count))} times)")
    print(f"  Sample: {last_frame_codes[:20]}")

    # Check for repeated patterns (silence indicator)
    print(f"\nChecking for repeated start patterns (potential silence):")
    repeat_count = 0
    for d in data[:50]:  # Check first 50
        codes = d.get('audio_codes', [])
        if codes and len(codes) > 3:
            # Check if first few frames are identical (silence)
            if codes[0] == codes[1] == codes[2]:
                repeat_count += 1

    print(f"  Samples with repeated first 3 frames: {repeat_count}/50")

analyze_codes("T:/Projects/Qwen3-TTS/train_with_codes.jsonl", "OLD DATA")
analyze_codes("T:/Projects/Qwen3-TTS/training_data/train_with_codes.jsonl", "NEW DATA")
