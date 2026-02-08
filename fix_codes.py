"""Strip silence frames from START only (not end)."""
import json

INPUT = "T:/Projects/Qwen3-TTS/training_data/train_with_codes.jsonl"
OUTPUT = "T:/Projects/Qwen3-TTS/training_data/train_with_codes_stripped2.jsonl"

# 300ms at 12Hz = ~3.6 frames, round to 4
FRAMES_TO_STRIP_START = 4
FRAMES_TO_STRIP_END = 0  # Don't strip from end

with open(INPUT, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} samples")
print(f"Stripping {FRAMES_TO_STRIP_START} frames from start only")

fixed = []
for d in data:
    codes = d.get('audio_codes', [])
    if len(codes) > FRAMES_TO_STRIP_START + 10:
        d['audio_codes'] = codes[FRAMES_TO_STRIP_START:]
        fixed.append(d)

print(f"Kept {len(fixed)} samples")

# Check variety
first_frames = [d['audio_codes'][0][0] for d in fixed]
print(f"First frame unique values: {len(set(first_frames))}")

# Fix paths to absolute
for d in fixed:
    ref = d.get('ref_audio', '')
    if not ref.startswith('T:'):
        d['ref_audio'] = 'T:/Projects/Qwen3-TTS/' + ref.replace('\\', '/')
    audio = d.get('audio', '')
    if not audio.startswith('T:'):
        d['audio'] = 'T:/Projects/Qwen3-TTS/' + audio.replace('\\', '/')

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for d in fixed:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f"Saved to: {OUTPUT}")
