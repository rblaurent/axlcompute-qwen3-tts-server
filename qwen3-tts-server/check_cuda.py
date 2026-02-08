"""Check CUDA usage and device info."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "qwen3-tts-repo"))

import torch

print("=" * 60)
print("CUDA AVAILABILITY CHECK")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    print(f"\nCurrent device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name()}")
else:
    print("\n[WARNING] CUDA is NOT available!")
    print("The model will run on CPU, which is much slower.")

# Check flash attention
print("\n" + "=" * 60)
print("FLASH ATTENTION CHECK")
print("=" * 60)

try:
    import flash_attn
    print(f"\nflash-attn installed: Yes")
    print(f"flash-attn version: {flash_attn.__version__}")
except ImportError:
    print(f"\nflash-attn installed: No")
    print("Flash Attention 2 is NOT installed.")
    print("This can provide 2-4x speedup for transformer attention.")

# Check if model uses CUDA
print("\n" + "=" * 60)
print("MODEL DEVICE CHECK")
print("=" * 60)

try:
    from qwen_tts import Qwen3TTSModel

    model_path = "T:/Projects/Qwen3-TTS/thalya/model/checkpoint"
    print(f"\nLoading model from: {model_path}")

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # Check where model components are
    print("\nModel component devices:")

    # Main model
    try:
        param = next(model.model.parameters())
        print(f"  Main model: {param.device}")
    except:
        print("  Main model: (could not determine)")

    # Speech tokenizer decoder
    try:
        decoder = model.model.speech_tokenizer.model.decoder
        param = next(decoder.parameters())
        print(f"  Decoder: {param.device}")
    except:
        print("  Decoder: (could not determine)")

    # Check memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

except Exception as e:
    print(f"\nError loading model: {e}")

print("\n" + "=" * 60)
print("SOX CHECK")
print("=" * 60)

import shutil
sox_path = shutil.which("sox")
if sox_path:
    print(f"\nSoX found at: {sox_path}")
else:
    print("\nSoX NOT found in PATH")
    print("SoX is used by some audio processing libraries (like pysox/sox).")
    print("However, this TTS system uses PyTorch for audio decoding,")
    print("so SoX is likely NOT critical for performance.")
