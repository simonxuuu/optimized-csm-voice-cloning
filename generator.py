from dataclasses import dataclass
from typing import List, Tuple

import os
import pickle

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        # Use torch.cuda.amp.autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        max_seq_len: int = 2048,
        batch_size: int = 16,  # Add batch processing
        apply_watermark: bool = False,  # Make watermarking optional
        optimize_memory: bool = True,   # Enable memory optimization
    ) -> torch.Tensor:
        self._model.reset_caches()

        # Enable CUDA graphs if available for faster CUDA kernel launches
        use_cuda_graphs = torch.cuda.is_available() and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'CUDAGraph')
        
        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device, non_blocking=True)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device, non_blocking=True)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = max_seq_len - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        # Pre-allocate tensors for the batch processing
        if optimize_memory:
            torch.cuda.empty_cache()
            
        # Process frames in batches for better efficiency
        for frame_idx in range(0, max_audio_frames, batch_size):
            # Calculate actual batch size for this iteration
            current_batch_size = min(batch_size, max_audio_frames - frame_idx)
            
            if current_batch_size <= 0:
                break
                
            # Use torch.cuda.amp.autocast for mixed precision computation
            with torch.cuda.amp.autocast(enabled=True):
                batch_samples = []
                for _ in range(current_batch_size):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    if torch.all(sample == 0):
                        break  # eos
                    
                    batch_samples.append(sample)
                    
                    # Update for next frame
                    curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1
                
                # If we hit EOS, stop processing
                if len(batch_samples) < current_batch_size:
                    samples.extend(batch_samples)
                    break
                    
                samples.extend(batch_samples)

        # Check if we have samples before proceeding
        if not samples:
            return torch.zeros(0).to(self.device)
            
        # Stack samples efficiently
        audio_tokens = torch.stack(samples).permute(1, 2, 0)
        
        # Use mixed precision for audio decoding
        with torch.cuda.amp.autocast(enabled=True):
            audio = self._audio_tokenizer.decode(audio_tokens).squeeze(0).squeeze(0)

        # Apply watermark only if requested
        if apply_watermark:
            # This applies an imperceptible watermark to identify audio as AI-generated.
            # Watermarking ensures transparency, dissuades misuse, and enables traceability.
            # Please be a responsible AI citizen and keep the watermarking in place.
            # If using CSM 1B in another application, use your own private key and keep it secret.
            audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio

    def save_cloned_voice(self, audio_tokens: torch.Tensor, file_path: str):
        """
        Save the cloned voice's audio tokens to a file for reuse.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(audio_tokens.cpu(), f)

    def load_cloned_voice(self, file_path: str) -> torch.Tensor:
        """
        Load the cloned voice's audio tokens from a file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cloned voice file not found: {file_path}")

        with open(file_path, 'rb') as f:
            audio_tokens = pickle.load(f)

        return audio_tokens.to(self.device)

    def generate_from_cloned_voice(
        self,
        text: str,
        speaker: int,
        cloned_voice_path: str,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        max_seq_len: int = 2048,
        batch_size: int = 16,
        apply_watermark: bool = False,
    ) -> torch.Tensor:
        """
        Generate audio using pre-saved cloned voice tokens.
        """
        self._model.reset_caches()

        # Load cloned voice tokens
        cloned_voice_tokens = self.load_cloned_voice(cloned_voice_path)

        # Tokenize the input text
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)

        # Combine cloned voice tokens with text tokens
        prompt_tokens = torch.cat([cloned_voice_tokens, gen_segment_tokens], dim=0).long().to(self.device, non_blocking=True)
        prompt_tokens_mask = torch.cat([
            torch.ones_like(cloned_voice_tokens, dtype=torch.bool),
            gen_segment_tokens_mask
        ], dim=0).bool().to(self.device, non_blocking=True)

        # Generate audio
        return self.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            apply_watermark=apply_watermark
        )


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    # Use mixed precision for model loading
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    
    # Load model with map_location to ensure proper device placement
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Enable torch.compile if available for model optimization
    if hasattr(torch, 'compile') and device == "cuda":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    generator = Generator(model)
    return generator
