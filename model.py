"""
Transformer-based denoising model for discrete diffusion.

Takes (corrupted token sequence, timestep) and predicts the original clean tokens.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from tokenizer import VOCAB_SIZE, PAD_ID, MASK_ID


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) -> (batch, dim)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DenoisingTransformer(nn.Module):
    """
    Encoder-only transformer that predicts original tokens from corrupted input.

    Input: (token_ids, timestep)
    Output: logits [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.timestep_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Token IDs [batch, seq_len]
            t: Timestep [batch] or [batch, 1], values in [0, T]
            pad_mask: True where padding (ignore). [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        B, S = x.shape
        token_ids = x
        x = self.token_embed(x) * math.sqrt(self.d_model)
        t_emb = self.timestep_embed(t.squeeze(-1) if t.dim() > 1 else t)
        x = x + t_emb.unsqueeze(1)
        x = x + self.pos_embed[:, :S]
        x = self.dropout(x)

        # Transformer key_padding_mask: True = ignore (padding)
        if pad_mask is None:
            pad_mask = (token_ids == PAD_ID)
        src_key_padding = pad_mask.bool()

        out = self.transformer(x, src_key_padding_mask=src_key_padding)
        return self.out_proj(out)
