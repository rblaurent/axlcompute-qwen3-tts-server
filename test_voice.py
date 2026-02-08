"""Test our custom voice with verbose English instructions + natural French text."""
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

print("Loading OUR fine-tuned custom voice...")
model = Qwen3TTSModel.from_pretrained(
    "T:/Projects/Qwen3-TTS/output_lowlr/checkpoint-epoch-5",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Natural French texts with speech markers
tests = [
    (
        "Hmm... ouais, je sais pas trop en fait. Genre, c'est compliqué quoi.",
        "Speak hesitantly, thinking aloud, unsure",
        "nat1_hesitant.wav"
    ),
    (
        "Pfff, mais attends attends, t'es sérieux là? Non mais c'est n'importe quoi!",
        "Speak with exasperation and disbelief, annoyed",
        "nat2_exasperated.wav"
    ),
    (
        "Ah! Mais oui! Mais c'est bien sûr! Du coup, voilà quoi, j'avais pas compris!",
        "Speak with sudden realization, excited and happy",
        "nat3_realization.wav"
    ),
    (
        "Euh... bon bah... comment dire... enfin bref, c'est pas grave.",
        "Speak awkwardly, searching for words, a bit embarrassed",
        "nat4_awkward.wav"
    ),
    (
        "Oh là là, mais c'est génial ça! Genre, vraiment vraiment top!",
        "Speak with enthusiasm and excitement, very happy",
        "nat5_excited.wav"
    ),
    (
        "Tss tss tss... non non non, ça va pas du tout. Tu vois ce que je veux dire?",
        "Speak disapprovingly, shaking head, concerned",
        "nat6_disapproval.wav"
    ),
    (
        "Bah écoute, honnêtement? Hmm... j'en sais rien du tout en fait.",
        "Speak frankly but uncertain, shrugging",
        "nat7_honest.wav"
    ),
    (
        "Ha ha! Non mais sérieux, c'est trop drôle! Genre, j'y crois pas!",
        "Speak while laughing, finding it hilarious, amused",
        "nat8_laughing.wav"
    ),
    (
        "Oh... euh... je... enfin... c'est triste quoi...",
        "Speak sadly, voice breaking, emotional",
        "nat9_sad.wav"
    ),
    (
        "Woh woh woh! Attends attends! Doucement! On se calme là, ok?",
        "Speak urgently, trying to stop someone, alarmed",
        "nat10_urgent.wav"
    ),
    (
        "Bon, alors, du coup, en fait, voilà quoi. C'est comme ça.",
        "Speak casually, wrapping up, matter-of-fact",
        "nat11_casual.wav"
    ),
    (
        "Mmh mmh, oui oui, je vois le truc. Ah ouais, d'accord d'accord.",
        "Speak while nodding along, understanding, agreeing",
        "nat12_agreeing.wav"
    ),
]

for text, instruct, filename in tests:
    print(f"\n--- {filename} ---")
    print(f"Text: {text[:60]}...")
    print(f"Style: {instruct[:50]}...")

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="French",
        speaker="maryam",
        instruct=instruct,
        max_new_tokens=2048,
    )

    duration = len(wavs[0]) / sr
    sf.write(filename, wavs[0], sr)
    print(f"Saved: {filename} ({duration:.1f}s)")

print("\n" + "="*50)
print("Done! Check nat*.wav files")
print("="*50)
