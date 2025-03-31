from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import pickle
import os
import time

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark

# Voice cache directory
VOICE_CACHE_DIR = "./voice_cache"
os.makedirs(VOICE_CACHE_DIR, exist_ok=True)

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
        
        # Voice cache storage
        self.voice_cache = {}

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

    def save_voice(self, voice_id: str, segment: Segment, filepath: Optional[str] = None) -> str:
        """
        Save a processed voice segment for later reuse.
        
        Args:
            voice_id: Identifier for the voice
            segment: The voice segment to save
            filepath: Optional custom path to save the voice data
            
        Returns:
            The path where the voice was saved
        """
        if filepath is None:
            # Create a standardized filename
            filepath = os.path.join(VOICE_CACHE_DIR, f"{voice_id}.voice.pkl")
            
        # Store in memory cache
        self.voice_cache[voice_id] = segment
        
        # Save to disk
        voice_data = {
            "segment": segment,
            "sample_rate": self.sample_rate,
            "created_at": time.time(),
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(voice_data, f)
            
        print(f"Voice '{voice_id}' saved to {filepath}")
        return filepath
    
    def load_voice(self, voice_id: str, filepath: Optional[str] = None) -> Optional[Segment]:
        """
        Load a voice from cache or file.
        
        Args:
            voice_id: Identifier for the voice
            filepath: Optional path to load from (otherwise uses standard path)
            
        Returns:
            The loaded voice segment or None if not found
        """
        # Check memory cache first
        if voice_id in self.voice_cache:
            return self.voice_cache[voice_id]
            
        # Determine filepath if not provided
        if filepath is None:
            filepath = os.path.join(VOICE_CACHE_DIR, f"{voice_id}.voice.pkl")
            
        # Load from disk if exists
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    voice_data = pickle.load(f)
                
                segment = voice_data["segment"]
                
                # Store in memory cache for future use
                self.voice_cache[voice_id] = segment
                
                print(f"Voice '{voice_id}' loaded from {filepath}")
                return segment
            except Exception as e:
                print(f"Error loading voice: {e}")
                return None
        else:
            print(f"Voice file not found: {filepath}")
            return None
    
    def list_cached_voices(self) -> List[str]:
        """
        List all available cached voices.
        
        Returns:
            List of voice IDs that are available
        """
        # List from memory cache
        memory_voices = list(self.voice_cache.keys())
        
        # List from disk cache
        disk_voices = []
        for filename in os.listdir(VOICE_CACHE_DIR):
            if filename.endswith(".voice.pkl"):
                voice_id = filename.split(".")[0]
                disk_voices.append(voice_id)
                
        # Combine and deduplicate
        all_voices = list(set(memory_voices + disk_voices))
        return all_voices
    
    def clone_and_save_voice(
        self, 
        audio_path: str, 
        voice_id: str, 
        reference_text: str = ""
    ) -> Optional[Segment]:
        """
        Process an audio file, create a voice segment, and save it for reuse.
        
        Args:
            audio_path: Path to the audio file to process
            voice_id: ID to give the processed voice
            reference_text: Optional reference text for the audio
            
        Returns:
            The processed voice segment or None if failed
        """
        try:
            # Load and process audio
            audio, sr = torchaudio.load(audio_path)
            audio = audio.mean(dim=0)  # Convert to mono
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=self.sample_rate
                )
            
            # Normalize audio volume
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Create segment
            segment = Segment(
                text=reference_text,
                speaker=999,
                audio=audio
            )
            
            # Save the voice
            self.save_voice(voice_id, segment)
            
            return segment
        except Exception as e:
            print(f"Error processing voice: {e}")
            return None

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
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = max_seq_len - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio

    @torch.inference_mode()
    def generate_with_cached_voice(
        self,
        text: str,
        voice_id: str,
        max_audio_length_ms: float = 5_000,
        temperature: float = 0.6,
        topk: int = 20,
        max_seq_len: int = 2048,
    ) -> torch.Tensor:
        """
        Generate speech using a cached voice.
        
        Args:
            text: The text to speak
            voice_id: Voice ID or filepath to use
            max_audio_length_ms: Maximum output audio length in ms
            temperature: Generation temperature (higher = more diverse)
            topk: Top-k sampling parameter
            max_seq_len: Maximum sequence length
            
        Returns:
            Generated audio tensor
        """
        # Load voice
        if os.path.exists(voice_id) and voice_id.endswith(".voice.pkl"):
            # If voice_id is actually a filepath
            filepath = voice_id
            voice_id = os.path.basename(filepath).split(".")[0]
            segment = self.load_voice(voice_id, filepath)
        else:
            # Try to load by ID
            segment = self.load_voice(voice_id)
            
        if segment is None:
            raise ValueError(f"Voice '{voice_id}' not found")
            
        # Generate with the loaded voice
        return self.generate(
            text=text,
            speaker=999,
            context=[segment],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            max_seq_len=max_seq_len
        )


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda", compile_model: bool = True, use_cached_compile: bool = True) -> Generator:
    """
    Load CSM 1B model with optional compilation.
    
    Args:
        ckpt_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        compile_model: Whether to compile the model with torch.compile
        use_cached_compile: Whether to load a previously cached compiled model (if available)
                            or save the newly compiled model for future use
    
    Returns:
        Generator instance
    """
    import os
    import hashlib
    
    # Generate a hash from the checkpoint path to use in the cache filename
    model_hash = hashlib.md5(ckpt_path.encode()).hexdigest()[:8]
    cached_model_path = f"./compiled_model_cache_{model_hash}.pt"
    
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Check if we should load a cached compiled model
    if compile_model and use_cached_compile and os.path.exists(cached_model_path):
        try:
            print(f"Loading cached compiled model from {cached_model_path}...")
            model = torch.load(cached_model_path, map_location=device)
            print("Cached compiled model loaded successfully.")
            model = model.to(device=device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Failed to load cached model: {e}")
            print("Falling back to regular model loading...")
            model = Model(model_args).to(device=device, dtype=torch.bfloat16)
            state_dict = torch.load(ckpt_path)
            model.load_state_dict(state_dict)
    else:
        # Load model normally
        model = Model(model_args).to(device=device, dtype=torch.bfloat16)
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
    
    # Apply torch.compile to optimize model execution if requested
    if compile_model and (not use_cached_compile or not os.path.exists(cached_model_path)):
        try:
            print("Compiling model with torch.compile()...")
            # Use fullgraph=True to prevent dtype issues
            # Set mode to 'reduce-overhead' to prevent precision changes
            compiled_model = torch.compile(
                model, 
                dynamic=True, 
                fullgraph=True, 
                mode='reduce-overhead',  # This preserves dtypes better
                backend='inductor'       # Explicitly use the default backend
            )
            
            # Save the compiled model for future use if caching is enabled
            if use_cached_compile:
                try:
                    print(f"Saving compiled model to {cached_model_path} for faster future loading...")
                    torch.save(compiled_model, cached_model_path)
                    print("Compiled model cached successfully.")
                except Exception as e:
                    print(f"Warning: Failed to save compiled model: {e}")
            
            model = compiled_model
            print("Model compilation complete.")
        except Exception as e:
            print(f"Warning: Model compilation failed with error: {e}")
            print("Continuing with uncompiled model.")

    generator = Generator(model)
    return generator