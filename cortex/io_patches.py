# cortex/io_patches.py
# Minimal text sensor that embeds tokens into d_model with a learned positional buffer.

import torch
import torch.nn as nn

class TextSensor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, ctx_len: int, tie_embedding: bool = True):
        super().__init__()
        self.ctx_len = ctx_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(ctx_len, d_model) / (d_model ** 0.5))
        self.tie_embedding = tie_embedding

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, T<=ctx_len]
        Returns: embeddings [B, T, D], and the embedding module for potential weight tying
        """
        B, T = tokens.shape
        x = self.emb(tokens) + self.pos[:T]  # [B, T, D]
        return x, (self.emb if self.tie_embedding else None)
