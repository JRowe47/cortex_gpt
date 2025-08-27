# sliding_window_ae.py
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from mfs import MFSHead

class SlidingWindowAE(nn.Module):
    """
    Lightweight L4-like autoencoder head:
      - Reconstruct x_t (categorical) and predict x_{t+1} using MFS heads.
      - Keeps a small state vector per stream (can be batched).
    Notes:
      * We must ignore index -1 in the next-step labels because the last
        position in each sequence has no next token by construction.
    """
    def __init__(self, d_model, vocab_size, mfs_k=3, mfs_p=4, slate_dim=0, low_rank_r=0, tie_weight=None):
        super().__init__()
        self.state_proj = nn.Linear(d_model, d_model)
        self.recon = MFSHead(
            d_model, vocab_size,
            K=mfs_k, P=mfs_p, slate_dim=slate_dim, low_rank_r=low_rank_r, tie_weight=tie_weight
        )
        self.nextp = MFSHead(
            d_model, vocab_size,
            K=mfs_k, P=mfs_p, slate_dim=slate_dim, low_rank_r=low_rank_r, tie_weight=tie_weight
        )

    def forward(self, h, slate, targets, next_targets, temperature=1.0, return_losses=False):
        """
        h:            [B,T,D] token features (pre‑head)
        slate:        [B,T,S] pooled state slate (aux conditioning)
        targets:      [B,T]   indices for x_t (reconstruction)
        next_targets: [B,T]   indices for x_{t+1}; last position should be -1 (ignore)
        return_losses: False -> return (logp_rec, logp_nxt)
                        True  -> return (loss_rec, loss_nxt) as scalars
        """
        # Heads return LOG-PROBS when return_logprobs=True
        logp_rec = self.recon(h, slate=slate, temperature=temperature, return_logprobs=True)  # [B,T,V]
        logp_nxt = self.nextp(h, slate=slate, temperature=temperature, return_logprobs=True) # [B,T,V]

        if not return_losses:
            return logp_rec, logp_nxt

        V = logp_rec.size(-1)
        loss_rec = F.nll_loss(
            logp_rec.view(-1, V), targets.view(-1),
            reduction='mean', ignore_index=-1  # tolerate masked labels if any
        )
        loss_nxt = F.nll_loss(
            logp_nxt.view(-1, V), next_targets.view(-1),
            reduction='mean', ignore_index=-1  # crucial: last position is -1
        )
        return loss_rec, loss_nxt