"""Fix paths in stripped jsonl to be absolute."""
import json

INPUT = "T:/Projects/Qwen3-TTS/training_data/train_with_codes_stripped.jsonl"

with open(INPUT, 'r') as f:
    data = [json.loads(l) for l in f]

# Fix paths to absolute
for d in data:
    ref = d.get('ref_audio', '')
    if not ref.startswith('T:'):
        d['ref_audio'] = 'T:/Projects/Qwen3-TTS/' + ref.replace('\\', '/')
    audio = d.get('audio', '')
    if not audio.startswith('T:'):
        d['audio'] = 'T:/Projects/Qwen3-TTS/' + audio.replace('\\', '/')

with open(INPUT, 'w') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'Fixed {len(data)} samples')
print(f'Sample ref_audio: {data[0].get("ref_audio")}')
print(f'Sample audio: {data[0].get("audio")}')
