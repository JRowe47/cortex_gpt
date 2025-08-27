# mfs.py
# 7-facet Mixture-of-Softmax head with optional top-m facet pruning and regularizers.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFacetSoftmax(nn.Module):
    """
    Mixture-of-Softmax head with K facets (default=7).
    - gate: produces facet weights per example
    - facets: K output projections to vocab
    Training extras:
      * entropy penalty on per-example gates (encourage sparse selections)
      * load-balance penalty on average gate usage across batch/time
    """
    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 num_facets: int = 7,
                 top_m: int | None = None,
                 temperature: float = 1.0,
                 entropy_coef: float = 1e-3,
                 balance_coef: float = 1e-2,
                 eps: float = 1e-9,
                 tie_to_embedding: nn.Embedding | None = None):
        super().__init__()
        self.K = num_facets
        self.top_m = top_m
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self.balance_coef = balance_coef
        self.eps = eps

        self.gate = nn.Linear(d_model, self.K)
        self.facets = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(self.K)
        ])

        if tie_to_embedding is not None:
            # Weight tying: decoder shares params with embedding.
            for f in self.facets:
                f.weight = tie_to_embedding.weight

    def forward(self, h: torch.Tensor, targets: torch.Tensor | None = None, reduce_ce: bool = True):
        """
        h: [B, D] or [B, T, D]
        targets: optional [B] or [B, T] token ids for CE loss
        Returns:
          If targets is None: (logp, aux)
          Else: (logp, loss, aux)
          where logp is log-prob over vocab with mixture, shape [B, V] or [B, T, V]
        """
        squeezed = (h.dim() == 2)
        if squeezed:
            h = h.unsqueeze(1)  # [B,1,D]
        B, T, D = h.shape

        # Gating
        gate_logits = self.gate(h)                # [B, T, K]
        gates = F.softmax(gate_logits, dim=-1)    # [B, T, K]

        # Optional facet pruning for compute (inference-time)
        if self.top_m is not None and self.top_m < self.K:
            topv, topi = gates.topk(self.top_m, dim=-1)                          # [B, T, m]
            gates_mask = torch.zeros_like(gates).scatter_(-1, topi, 1.0)
            gates = (gates * gates_mask)
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(self.eps)      # renorm

        # Per-facet probabilities (temperature scaled)
        ps = []
        for k in range(self.K):
            logits_k = self.facets[k](h) / self.temperature   # [B, T, V]
            ps.append(F.softmax(logits_k, dim=-1))            # [B, T, V]

        P = torch.stack(ps, dim=-2)                           # [B, T, K, V]
        mixP = (gates.unsqueeze(-1) * P).sum(dim=-2)          # [B, T, V]
        logp = torch.log(mixP.clamp_min(self.eps))            # [B, T, V]

        # Aux losses
        aux = {}
        # Encourage per-example sparsity (lower entropy)
        gate_entropy = -(gates.clamp_min(self.eps) * torch.log(gates.clamp_min(self.eps))).sum(-1)  # [B, T]
        aux['gate_entropy'] = gate_entropy.mean()
        # Global load balance across facets
        usage = gates.mean(dim=(0, 1))  # [K]
        aux['load_balance'] = ((usage - 1.0 / self.K) ** 2).mean()

        loss = None
        if targets is not None:
            targets = targets if targets.dim() == 2 else targets.unsqueeze(1)  # [B, T]
            nll = F.nll_loss(logp.reshape(-1, logp.size(-1)), targets.reshape(-1), reduction='mean')
            loss = nll + self.entropy_coef * aux['gate_entropy'] + self.balance_coef * aux['load_balance']

        if squeezed:
            logp = logp.squeeze(1)

        return (logp, loss, aux) if targets is not None else (logp, aux)


class MFSHead(nn.Module):
    """Compatibility wrapper around :class:`MultiFacetSoftmax`.

    The original project referenced an ``MFSHead`` class which is no longer
    present.  Some modules still import ``MFSHead`` from ``mfs`` and expect an
    interface that accepts a ``slate`` conditioning vector and returns
    log‑probabilities when requested.  This light wrapper delegates to
    :class:`MultiFacetSoftmax` and ignores extra arguments so that legacy code
    continues to run.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        K: int = 7,
        P: int = 1,
        slate_dim: int = 0,
        low_rank_r: int = 0,
        tie_weight: nn.Embedding | None = None,
    ):
        super().__init__()
        # ``P``, ``slate_dim`` and ``low_rank_r`` are accepted for backwards
        # compatibility but are not used by this simplified wrapper.
        self.mfs = MultiFacetSoftmax(
            d_model,
            vocab_size,
            num_facets=K,
        )
        if tie_weight is not None:
            weight = (
                tie_weight.weight if isinstance(tie_weight, nn.Embedding) else tie_weight
            )
            for f in self.mfs.facets:
                f.weight = weight

    def forward(
        self,
        h: torch.Tensor,
        slate: torch.Tensor | None = None,
        temperature: float = 1.0,
        return_logprobs: bool = False,
    ):
        # Temporarily override the temperature of the wrapped head.
        prev_temp = self.mfs.temperature
        self.mfs.temperature = temperature
        logp, _ = self.mfs(h)
        self.mfs.temperature = prev_temp

        if return_logprobs:
            return logp
        else:
            return logp.exp()


# Allow importing with either ``MFSHead`` or ``MFShead`` (historic casing).
MFShead = MFSHead

