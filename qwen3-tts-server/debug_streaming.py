"""Debug script to understand why streaming callbacks aren't working."""

import time
import torch

from qwen_tts import Qwen3TTSModel

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Patch the talker's forward to add debug output
original_forward = model.model.talker.forward

def debug_forward(*args, **kwargs):
    result = original_forward(*args, **kwargs)

    # Check if _last_codec_ids was set
    if hasattr(model.model.talker, '_last_codec_ids'):
        codec_ids = model.model.talker._last_codec_ids
        if codec_ids is not None:
            print(f"  [forward] _last_codec_ids set: shape={codec_ids.shape}")
        else:
            print(f"  [forward] _last_codec_ids is None")
    else:
        print(f"  [forward] _last_codec_ids not set as attribute")

    return result

model.model.talker.forward = debug_forward

# Create a test streamer to see if put() is called
from transformers import BaseStreamer

class DebugStreamer(BaseStreamer):
    def __init__(self, talker):
        self.talker = talker
        self.call_count = 0

    def put(self, value):
        self.call_count += 1
        print(f"  [streamer.put] call #{self.call_count}, value shape: {value.shape if hasattr(value, 'shape') else type(value)}")

        # Check _last_codec_ids
        if hasattr(self.talker, '_last_codec_ids'):
            codec_ids = self.talker._last_codec_ids
            if codec_ids is not None:
                print(f"    -> _last_codec_ids available: shape={codec_ids.shape}")
            else:
                print(f"    -> _last_codec_ids is None")
        else:
            print(f"    -> _last_codec_ids attribute doesn't exist")

    def end(self):
        print(f"  [streamer.end] called after {self.call_count} puts")

# Test direct talker.generate() with streamer
print("\n=== Test: Direct talker.generate() with streamer ===")

# We need to prepare inputs the same way the main generate() does
text = "Hello world."
input_ids = model._tokenize_texts([model._build_assistant_text(text)])

# Minimal setup - let's just see if the streamer gets called at all
# We'll call talker.generate directly with a simple input

# First, let's understand what inputs the talker expects
print(f"Talker type: {type(model.model.talker)}")
print(f"Talker config: {model.model.talker.config}")

# Create dummy inputs for talker
batch_size = 1
seq_len = 10
hidden_size = model.model.talker.config.hidden_size

dummy_embeds = torch.randn(batch_size, seq_len, hidden_size,
                           device=model.model.talker.device,
                           dtype=torch.bfloat16)
dummy_mask = torch.ones(batch_size, seq_len,
                        device=model.model.talker.device,
                        dtype=torch.long)

# Trailing text hidden and tts_pad_embed are needed
trailing_hidden = torch.randn(batch_size, 5, hidden_size,
                              device=model.model.talker.device,
                              dtype=torch.bfloat16)
tts_pad_embed = torch.randn(1, 1, hidden_size,
                            device=model.model.talker.device,
                            dtype=torch.bfloat16)

print("\nCalling talker.generate() with DebugStreamer...")
streamer = DebugStreamer(model.model.talker)

try:
    result = model.model.talker.generate(
        inputs_embeds=dummy_embeds,
        attention_mask=dummy_mask,
        trailing_text_hidden=trailing_hidden,
        tts_pad_embed=tts_pad_embed,
        max_new_tokens=10,
        min_new_tokens=2,
        do_sample=True,
        streamer=streamer,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    print(f"\nGeneration completed. Streamer was called {streamer.call_count} times.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Now test via model.model.generate() ===")

callback_count = [0]

def my_callback(codec_ids):
    callback_count[0] += 1
    print(f"  [callback] #{callback_count[0]}, codec_ids shape: {codec_ids.shape}")

print(f"Generating with codec_callback...")
try:
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=[None],
        languages=["English"],
        speakers=["Serena"],
        non_streaming_mode=True,
        codec_callback=my_callback,
        max_new_tokens=20,
    )
    print(f"\nGeneration completed.")
    print(f"Callback was called {callback_count[0]} times.")
    print(f"Output tokens: {talker_codes_list[0].shape if talker_codes_list else 'None'}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
