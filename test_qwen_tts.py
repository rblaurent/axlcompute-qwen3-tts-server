"""
Quick test script for Qwen3-TTS
Tests: model loading, custom voice generation, and voice cloning
"""
import torch
import soundfile as sf
from pathlib import Path

print("=" * 50)
print("Qwen3-TTS Local Test")
print("=" * 50)

# Check CUDA availability
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import qwen_tts
print("\nImporting qwen_tts...")
from qwen_tts import Qwen3TTSModel

# Test 1: Load CustomVoice model and generate speech
print("\n" + "=" * 50)
print("Test 1: CustomVoice Model (with style instructions)")
print("=" * 50)

print("\nLoading Qwen3-TTS-12Hz-1.7B-CustomVoice model...")
print("(This will download ~3.5GB on first run)")

try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")

    # Get available speakers and languages
    speakers = model.get_supported_speakers()
    languages = model.get_supported_languages()
    print(f"\nAvailable speakers: {speakers}")
    print(f"Available languages: {languages}")

    # Generate French speech with style instruction
    print("\nGenerating French speech with emotion...")
    text_fr = "Bonjour! Comment allez-vous aujourd'hui? Je suis vraiment content de vous voir!"

    wavs, sr = model.generate_custom_voice(
        text=text_fr,
        language="French",
        speaker="Ryan",  # English speaker but can do French
        instruct="Parle avec enthousiasme et joie, voix chaleureuse"
    )

    output_path = Path("test_output_french_custom.wav")
    sf.write(output_path, wavs[0], sr)
    print(f"Saved to: {output_path.absolute()}")
    print("SUCCESS: CustomVoice model works!")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Cleanup to free VRAM before next test
del model
torch.cuda.empty_cache()

# Test 2: Voice Clone model
print("\n" + "=" * 50)
print("Test 2: Base Model (Voice Cloning)")
print("=" * 50)

print("\nLoading Qwen3-TTS-12Hz-1.7B-Base model...")

try:
    clone_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")
    print("\nVoice cloning requires a reference audio file.")
    print("To test: provide a 3+ second WAV file of your voice with transcript.")
    print("SUCCESS: Base model loads correctly!")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("Tests complete!")
print("=" * 50)
