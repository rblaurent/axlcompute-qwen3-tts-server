"""
True Streaming TTS Generator for Qwen3-TTS (v2)

This module implements true token-level streaming by:
1. Preparing embeddings like the original generate() function
2. Running a custom generation loop that yields after each token
3. Decoding audio packets incrementally as tokens are produced

The key insight is that we need to bypass model.generate() and implement
our own generation loop to get access to intermediate codec tokens.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    packet_size: int = 4  # Tokens per packet (4 = ~320ms audio)
    left_context: int = 25  # Context tokens for smooth decode
    sample_rate: int = 24000
    max_new_tokens: int = 4096
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05


@dataclass
class GenerationState:
    """State for incremental generation."""
    past_key_values: Optional[Tuple] = None
    past_hidden: Optional[torch.Tensor] = None
    generation_step: int = 0
    generated_tokens: List[torch.Tensor] = field(default_factory=list)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a token from logits with temperature, top-k, and top-p."""
    if temperature > 0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


class TrueStreamingGenerator:
    """
    True token-level streaming for Qwen3-TTS.

    This generator implements its own token-by-token generation loop,
    allowing us to yield audio packets as tokens are generated.
    """

    def __init__(self, model, config: Optional[StreamingConfig] = None):
        self.model = model
        self.config = config or StreamingConfig()

        # Get references to model components
        self.tts_model = model.model  # Qwen3TTSForConditionalGeneration
        self.talker = self.tts_model.talker
        self.decoder = model.model.speech_tokenizer.model.decoder
        self.upsample_rate = self.decoder.total_upsample

        # Get special token IDs
        self.codec_eos_id = self.tts_model.config.talker_config.codec_eos_token_id
        self.codec_pad_id = self.tts_model.config.talker_config.codec_pad_id
        self.codec_bos_id = self.tts_model.config.talker_config.codec_bos_id

    def _prepare_inputs(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for generation (mirrors the logic in generate()).

        Returns:
            (input_embeds, trailing_text_hidden, tts_pad_embed)
        """
        # Tokenize text
        input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])
        input_id = input_ids[0]

        # Tokenize instruction if provided
        instruct_embed = None
        if instruct:
            instruct_ids = self.model._tokenize_texts([self.model._build_instruct_text(instruct)])
            instruct_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(instruct_ids[0])
            )

        # Get language ID
        if language.lower() == "auto":
            language_id = None
        else:
            language_id = self.tts_model.config.talker_config.codec_language_id.get(
                language.lower()
            )

        # Get speaker embed
        if speaker.lower() in self.tts_model.config.talker_config.spk_id:
            spk_id = self.tts_model.config.talker_config.spk_id[speaker.lower()]
            speaker_embed = self.talker.get_input_embeddings()(
                torch.tensor(spk_id, device=self.talker.device, dtype=input_id.dtype)
            )
        else:
            speaker_embed = None

        # Build TTS special embeddings
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.tts_model.config.tts_bos_token_id,
                      self.tts_model.config.tts_eos_token_id,
                      self.tts_model.config.tts_pad_token_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        # Build codec prefill sequence
        if language_id is None:
            codec_prefill_list = [[
                self.tts_model.config.talker_config.codec_nothink_id,
                self.tts_model.config.talker_config.codec_think_bos_id,
                self.tts_model.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.tts_model.config.talker_config.codec_think_id,
                self.tts_model.config.talker_config.codec_think_bos_id,
                language_id,
                self.tts_model.config.talker_config.codec_think_eos_id,
            ]]

        codec_input_emb_0 = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=self.talker.device, dtype=input_id.dtype)
        )
        codec_input_emb_1 = self.talker.get_input_embeddings()(
            torch.tensor(
                [[self.codec_pad_id, self.codec_bos_id]],
                device=self.talker.device,
                dtype=input_id.dtype,
            )
        )

        if speaker_embed is None:
            codec_input_emb = torch.cat([codec_input_emb_0, codec_input_emb_1], dim=1)
        else:
            codec_input_emb = torch.cat([
                codec_input_emb_0,
                speaker_embed.view(1, 1, -1),
                codec_input_emb_1
            ], dim=1)

        # Build input embeddings
        # <|im_start|>assistant\n
        role_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_id[:, :3])
        )

        # tts_pad * N + tts_bos + codec_input
        input_embed = torch.cat((
            tts_pad_embed.expand(-1, codec_input_emb.shape[1] - 2, -1),
            tts_bos_embed,
        ), dim=1) + codec_input_emb[:, :-1]

        talker_input_embed = torch.cat((role_embed, input_embed), dim=1)

        # Add first text token
        talker_input_embed = torch.cat([
            talker_input_embed,
            self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, 3:4])
            ) + codec_input_emb[:, -1:]
        ], dim=1)

        # Build trailing text hidden (remaining text to be consumed during generation)
        trailing_text_hidden = torch.cat((
            self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, 4:-5])
            ),
            tts_eos_embed
        ), dim=1)

        # Add instruction embedding if present
        if instruct_embed is not None:
            talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _generate_token(
        self,
        state: GenerationState,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, GenerationState]:
        """
        Generate a single token and update state.

        Returns:
            (codec_ids, updated_state) where codec_ids has shape (1, num_codebooks)
        """
        with torch.no_grad():
            # Call forward
            outputs = self.talker(
                input_ids=input_ids,
                inputs_embeds=input_embeds if input_ids is None else None,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                past_hidden=state.past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=state.generation_step,
                use_cache=True,
                output_hidden_states=True,
                subtalker_dosample=True,
                subtalker_top_k=self.config.top_k,
                subtalker_top_p=self.config.top_p,
                subtalker_temperature=self.config.temperature,
            )

            # Sample next token from logits
            logits = outputs.logits[:, -1, :]
            next_token = sample_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

            # Get codec_ids from hidden_states (all 16 codebooks)
            codec_ids = outputs.hidden_states[1]  # (hidden_states, codec_ids) tuple

            # Update state
            new_state = GenerationState(
                past_key_values=outputs.past_key_values,
                past_hidden=outputs.past_hidden,
                generation_step=outputs.generation_step,
                generated_tokens=state.generated_tokens + [codec_ids],
            )

            return next_token, codec_ids, new_state

    def _decode_packet(
        self,
        tokens: List[torch.Tensor],
        context_tokens: List[torch.Tensor],
    ) -> np.ndarray:
        """Decode a packet of tokens to audio."""
        # Combine context and new tokens
        all_tokens = context_tokens + tokens

        # Stack: list of (1, num_codebooks) -> (1, seq_len, num_codebooks)
        stacked = torch.cat(all_tokens, dim=0).unsqueeze(0)

        # Transpose to (1, num_codebooks, seq_len) for decoder
        codes = stacked.permute(0, 2, 1).to(self.decoder.device)

        with torch.no_grad():
            wav = self.decoder(codes)

        # Trim context
        context_samples = len(context_tokens) * self.upsample_rate
        wav_trimmed = wav[..., context_samples:]

        return wav_trimmed.squeeze().float().cpu().numpy()

    async def generate_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio with true token-level streaming.

        Yields audio packets as tokens are generated.
        """
        # Prepare inputs
        input_embeds, trailing_text_hidden, tts_pad_embed = self._prepare_inputs(
            text, speaker, language, instruct
        )

        # Initial attention mask
        attention_mask = torch.ones(
            (1, input_embeds.shape[1]),
            device=self.talker.device,
            dtype=torch.long,
        )

        # Initialize state
        state = GenerationState()

        # Token buffer for packet accumulation
        token_buffer: List[torch.Tensor] = []
        context_buffer: List[torch.Tensor] = []

        # Generation loop
        start_time = time.perf_counter()
        first_packet_time = None
        is_prefill = True
        current_input = input_embeds
        current_token = None

        for step in range(self.config.max_new_tokens):
            # Generate one token
            if is_prefill:
                next_token, codec_ids, state = self._generate_token(
                    state, current_input, attention_mask,
                    trailing_text_hidden, tts_pad_embed,
                )
                is_prefill = False
            else:
                # Update attention mask for new token
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.talker.device, dtype=torch.long)
                ], dim=1)

                next_token, codec_ids, state = self._generate_token(
                    state, None, attention_mask,
                    trailing_text_hidden, tts_pad_embed,
                    input_ids=current_token,
                )

            current_token = next_token

            # Check for EOS
            if next_token.item() == self.codec_eos_id:
                break

            # Add to buffer
            token_buffer.append(codec_ids)

            # Check if we have a full packet
            if len(token_buffer) >= self.config.packet_size:
                # Decode packet
                packet_tokens = token_buffer[:self.config.packet_size]
                token_buffer = token_buffer[self.config.packet_size:]

                # Get context
                context = context_buffer[-self.config.left_context:] if context_buffer else []

                # Decode
                audio = self._decode_packet(packet_tokens, context)

                # Update context
                context_buffer.extend(packet_tokens)
                if len(context_buffer) > self.config.left_context * 2:
                    context_buffer = context_buffer[-self.config.left_context * 2:]

                if first_packet_time is None:
                    first_packet_time = time.perf_counter() - start_time
                    print(f"[TrueStream] First packet at {first_packet_time*1000:.0f}ms")

                # Convert to PCM and yield
                audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                yield audio_int16.tobytes()

                # Small yield to allow other async tasks
                await asyncio.sleep(0)

        # Flush remaining tokens
        if token_buffer:
            context = context_buffer[-self.config.left_context:] if context_buffer else []
            audio = self._decode_packet(token_buffer, context)
            audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            yield audio_int16.tobytes()

        total_time = time.perf_counter() - start_time
        print(f"[TrueStream] Complete: {state.generation_step} tokens in {total_time*1000:.0f}ms")


def audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert audio array to PCM bytes (s16le)."""
    audio = np.clip(audio.astype(np.float32), -1, 1)
    return (audio * 32767).astype(np.int16).tobytes()
