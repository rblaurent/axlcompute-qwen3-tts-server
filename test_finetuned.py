"""
Test the fine-tuned Qwen3-TTS model with custom speaker and style instructions.
"""
import torch
import soundfile as sf

print("Loading fine-tuned model...")
from qwen_tts import Qwen3TTSModel

# Load the latest checkpoint
model = Qwen3TTSModel.from_pretrained(
    "output_final/checkpoint-epoch-30",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print(f"Available speakers: {model.get_supported_speakers()}")
print(f"Available languages: {model.get_supported_languages()}")

# Test 1: French with enthusiastic style
print("\n--- Test 1: French with enthusiastic style ---")
text1 = "Bonjour! Aujourd'hui je vais vous expliquer comment atteindre vos objectifs!"
instruct1 = "Parle avec enthousiasme et énergie, voix motivante"

wavs1, sr = model.generate_custom_voice(
    text=text1,
    language="French",
    speaker="maryam",
    instruct=instruct1
)
sf.write("test_finetuned_enthusiastic.wav", wavs1[0], sr)
print(f"Generated: test_finetuned_enthusiastic.wav")
print(f"Text: {text1}")
print(f"Style: {instruct1}")

# Test 2: French with calm/serious style
print("\n--- Test 2: French with calm style ---")
text2 = "Il est important de comprendre que le succès demande de la patience et de la persévérance."
instruct2 = "Parle calmement et sérieusement, voix posée"

wavs2, sr = model.generate_custom_voice(
    text=text2,
    language="French",
    speaker="maryam",
    instruct=instruct2
)
sf.write("test_finetuned_calm.wav", wavs2[0], sr)
print(f"Generated: test_finetuned_calm.wav")
print(f"Text: {text2}")
print(f"Style: {instruct2}")

# Test 3: French with emotional/inspiring style
print("\n--- Test 3: French with inspiring style ---")
text3 = "Tu peux y arriver! Crois en toi et ne laisse personne te dire que c'est impossible!"
instruct3 = "Parle avec passion et conviction, voix inspirante et encourageante"

wavs3, sr = model.generate_custom_voice(
    text=text3,
    language="French",
    speaker="maryam",
    instruct=instruct3
)
sf.write("test_finetuned_inspiring.wav", wavs3[0], sr)
print(f"Generated: test_finetuned_inspiring.wav")
print(f"Text: {text3}")
print(f"Style: {instruct3}")

print("\n" + "="*60)
print("Fine-tuning test complete!")
print("Generated files:")
print("  - test_finetuned_enthusiastic.wav")
print("  - test_finetuned_calm.wav")
print("  - test_finetuned_inspiring.wav")
print("="*60)
