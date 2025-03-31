from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    # Create mask once and cache it
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  
    # More efficient multinomial sampling - optimized version
    with torch.no_grad():
        # Using exponential distribution for efficient gumbel-max sampling
        q = torch.empty_like(probs).exponential_(1.0)
        return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    with torch.no_grad():  # Ensure no gradients are computed
        # Apply temperature scaling
        if temperature > 0:
            logits = logits / temperature
        
        # Optimized top-k filtering
        if topk > 0 and topk < logits.size(-1):
            v, _ = torch.topk(logits, topk)
            logits[logits < v[..., -1, None]] = float('-inf')
        
        # More efficient softmax calculation
        scores_processed = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(scores_processed, dim=-1)
        
        # Sample from the distribution
        sample_token = _multinomial_sample_one_no_sync(probs)
        return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())

        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        # Fix: Explicitly ensure the model, cache, and any inputs share the same dtype
        self.model_dtype = dtype  # Save for reference 

        with device:
            # Force all caches to use bfloat16 if that's what the model is using
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

        # Cache masks for reuse
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device))
        
        # Pre-compute variables that will be used repeatedly
        self.vocab_size_offsets = torch.arange(
            0, self.args.audio_num_codebooks * self.args.audio_vocab_size, 
            self.args.audio_vocab_size, device=device
        )

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1) or (batch_size, seq_len)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1) or (batch_size, seq_len)
            input_pos: (batch_size, seq_len) positions for each token
            
        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype  # Ensure consistent dtype
        
        # Reshape tokens to have 3 dimensions if it doesn't already
        if tokens.dim() == 2:
            # If tokens is 2D (batch_size, seq_len), reshape to 3D
            tokens = tokens.unsqueeze(-1)
            
            # Also reshape tokens_mask if it has the same issue
            if tokens_mask.dim() == 2:
                tokens_mask = tokens_mask.unsqueeze(-1)
                
            # Expand to full size (batch_size, seq_len, audio_num_codebooks+1)
            tokens = tokens.expand(-1, -1, self.args.audio_num_codebooks+1)
            tokens_mask = tokens_mask.expand(-1, -1, self.args.audio_num_codebooks+1)
        
        # Now tokens should have 3 dimensions
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        
        # Ensure all tensors have the same dtype before processing
        embeds = self._embed_tokens(tokens).to(dtype=dtype)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1).to(dtype=dtype)
        h = masked_embeds.sum(dim=2).to(dtype=dtype)  # Ensure dtype consistency
        
        # Explicitly disable autocast for backbone to avoid dtype changes
        # This is the key fix for the KV cache dtype mismatch
        with torch.autocast(device_type='cuda', enabled=False):
            h = self.backbone(h.to(dtype=dtype), input_pos=input_pos, mask=curr_backbone_mask)
            h = h.to(dtype=dtype)  # Enforce consistency

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample).to(dtype=dtype)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        
        # Explicitly disable autocast for decoder to avoid dtype changes
        with torch.autocast(device_type='cuda', enabled=False):
            for i in range(1, self.args.audio_num_codebooks):
                curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
                
                # Ensure consistent dtype for each operation
                projection_out = self.projection(curr_h.to(dtype=dtype)).to(dtype=dtype)
                decoder_h = self.decoder(projection_out, input_pos=curr_pos, mask=curr_decoder_mask)
                decoder_h = decoder_h.to(dtype=dtype)
                
                ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
                ci_sample = sample_topk(ci_logits, topk, temperature)
                ci_embed = self._embed_audio(i, ci_sample).to(dtype=dtype)

                curr_h = ci_embed
                curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        # Use pre-computed offsets for efficiency
        if hasattr(self, 'vocab_size_offsets'):
            return self.audio_embeddings(tokens + self.vocab_size_offsets[codebook])
        else:
            return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Optimize embedding operations
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        
        # More efficient audio token processing
        if hasattr(self, 'vocab_size_offsets'):
            offsets = self.vocab_size_offsets
        else:
            offsets = torch.arange(self.args.audio_num_codebooks, device=tokens.device) * self.args.audio_vocab_size
            
        # Get audio tokens with broadcasting
        audio_tokens = tokens[:, :, :-1] + offsets
        
        # More efficient reshape operation
        batch_size, seq_len = tokens.shape[:2]
        audio_embeds = self.audio_embeddings(audio_tokens.reshape(-1)).reshape(
            batch_size, seq_len, self.args.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
