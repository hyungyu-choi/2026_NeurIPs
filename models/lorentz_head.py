# coding=utf-8
"""
Lorentz Score Head for Plackett-Luce temporal ordering on the hyperboloid.

Design principles
─────────────────
The ViT backbone + MERUStyleProjection places frame embeddings on the Lorentz
hyperboloid (only space components stored; time computed on-the-fly).  A naive
Euclidean Transformer head would destroy the geometric structure.  Instead we
use a "Tangent-Lorentz" architecture:

  1. **LorentzAttention**
     • Q/K are projected in the tangent space at origin (log_map0), then lifted
       back to the hyperboloid (exp_map0).
     • Attention logits are the *negated Lorentzian inner product* of Q and K,
       ⟨q, k⟩_L = q_space · k_space − q_time · k_time.
       On the same sheet of the hyperboloid this is always ≤ −1/c, so negating
       it gives large positive values for nearby points → natural soft-nearest-
       neighbour attention that respects hyperbolic distances.
     • V is kept in the tangent space (where weighted averaging = Fréchet-mean
       first-order approximation is valid).

  2. **LorentzMLP**
     • Operates entirely in the tangent space at origin: log_map0 → FC → GELU → FC.
     • The tangent space at the origin of the Lorentz model *is* Euclidean ℝ^D,
       so standard linear layers are mathematically consistent.

  3. **Residual connections**
     • Pre-norm (LayerNorm in tangent space).
     • Residual addition happens in tangent space:
         h_new = exp_map0( log_map0(h) + Δ,  curv )
       This is the "first-order retraction" commonly used in Riemannian
       optimisation and is exact when Δ is small.

  4. **Scoring**
     • After N Lorentz blocks the refined embeddings are mapped to tangent space
       one last time, normalised, and projected to a scalar via a small MLP.
     • This gives per-frame scores for the Plackett-Luce loss.
     • The intermediate hyperbolic embeddings are *also* returned so that the
       entailment-cone loss can be applied jointly.

Reference geometry (Lorentz / hyperboloid model):
    Hyperboloid:  ⟨x,x⟩_L = −1/c   where  ⟨x,y⟩_L = x_s·y_s − x_t·y_t
    x_time = sqrt(1/c + ||x_space||²)
    Only space components x_space ∈ ℝ^D are stored.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from models import lorentz_ops as L


# ═════════════════════════════════════════════
# Building blocks
# ═════════════════════════════════════════════

class LorentzAttention(nn.Module):
    """
    Multi-head attention whose logits are Lorentzian inner products.

    Flow:
        h (hyperboloid) ──log_map0──▶ h_tan (tangent)
            ├─ Wq ──▶ q_tan ──exp_map0──▶ q_hyp ─┐
            ├─ Wk ──▶ k_tan ──exp_map0──▶ k_hyp ─┤  ⟨q,k⟩_L
            └─ Wv ──▶ v_tan ─────────────────────┘  → softmax → aggregate v
        output: Wo(aggregated v_tan)   [still tangent space]
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(self, h: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:    (B, T, D)  space components on hyperboloid
            curv: positive scalar (curvature magnitude)
        Returns:
            (B, T, D) tangent-space vector (delta to be added as residual)
        """
        B, T, D = h.shape
        hd = self.head_dim

        # ── tangent space ──
        h_tan = L.log_map0(h.reshape(B * T, D), curv).reshape(B, T, D)

        q_tan = self.wq(h_tan).view(B, T, self.n_heads, hd).permute(0, 2, 1, 3)
        k_tan = self.wk(h_tan).view(B, T, self.n_heads, hd).permute(0, 2, 1, 3)
        v_tan = self.wv(h_tan).view(B, T, self.n_heads, hd).permute(0, 2, 1, 3)
        # shapes: (B, n_heads, T, hd)

        # ── lift Q, K back to hyperboloid for Lorentzian scoring ──
        q_hyp = L.exp_map0(q_tan.reshape(-1, hd), curv).view(B, self.n_heads, T, hd)
        k_hyp = L.exp_map0(k_tan.reshape(-1, hd), curv).view(B, self.n_heads, T, hd)

        # time components  (B, nH, T, 1)
        q_time = torch.sqrt(1.0 / curv + (q_hyp ** 2).sum(-1, keepdim=True))
        k_time = torch.sqrt(1.0 / curv + (k_hyp ** 2).sum(-1, keepdim=True))

        # Lorentzian inner product:  ⟨q,k⟩_L = q_s·k_s − q_t·k_t   ≤ −1/c
        # (B, nH, T, T)
        space_dot = torch.matmul(q_hyp, k_hyp.transpose(-1, -2))
        time_dot  = torch.matmul(q_time, k_time.transpose(-1, -2))
        lorentz_ip = space_dot - time_dot          # negative on same sheet

        # Negate → large positive for nearby pairs, then scale
        attn_logits = -lorentz_ip / math.sqrt(self.head_dim)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # ── aggregate values in tangent space (valid linear combination) ──
        out = torch.matmul(attn_weights, v_tan)    # (B, nH, T, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.wo(out)

        return out                                  # tangent-space delta


class LorentzMLP(nn.Module):
    """Feed-forward network operating in the tangent space at origin."""

    def __init__(self, embed_dim: int, hidden_mul: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(embed_dim * hidden_mul)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) tangent-space vectors."""
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LorentzBlock(nn.Module):
    """
    Pre-norm Transformer block on the Lorentz hyperboloid.

    Input/output live on the hyperboloid (space components only).
    Internally uses log/exp maps to bridge tangent ↔ manifold.
    """

    def __init__(self, embed_dim: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = LorentzAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = LorentzMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, h: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        """
        h:    (B, T, D)  space components on hyperboloid
        curv: positive scalar
        Returns: (B, T, D) space components on hyperboloid
        """
        B, T, D = h.shape

        # ── Self-attention sub-block ──
        h_tan = L.log_map0(h.reshape(B * T, D), curv).reshape(B, T, D)
        h_normed = self.norm1(h_tan)

        # Attention operates on the original hyperboloid embeddings
        # but we pass h (pre-residual) so it can compute Lorentzian distances
        attn_delta = self.attn(h, curv)             # tangent delta

        # Residual in tangent space → retract to hyperboloid
        h_new_tan = h_tan + attn_delta
        h = L.exp_map0(h_new_tan.reshape(B * T, D), curv).reshape(B, T, D)

        # ── MLP sub-block ──
        h_tan = L.log_map0(h.reshape(B * T, D), curv).reshape(B, T, D)
        h_normed = self.norm2(h_tan)

        mlp_delta = self.mlp(h_normed)              # tangent delta

        h_new_tan = h_tan + mlp_delta
        h = L.exp_map0(h_new_tan.reshape(B * T, D), curv).reshape(B, T, D)

        return h


# ═════════════════════════════════════════════
# Full Score Head
# ═════════════════════════════════════════════

class LorentzScoreHead(nn.Module):
    """
    Stack of LorentzBlocks → scalar score per frame.

    Returns
    -------
    scores : (B, T)   for Plackett-Luce loss
    h_out  : (B, T, D) refined space components (for entailment loss)
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int  = 2,
        n_heads: int   = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            LorentzBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Scorer: tangent space → scalar
        hidden = max(embed_dim // 2, 16)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, h: torch.Tensor, curv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:    (B, T, D) space components on hyperboloid
            curv: positive curvature scalar
        Returns:
            scores: (B, T)
            h:      (B, T, D) refined embeddings (still on hyperboloid)
        """
        B, T, D = h.shape

        for block in self.blocks:
            h = block(h, curv)

        # Map to tangent for scoring
        h_tan = L.log_map0(h.reshape(B * T, D), curv).reshape(B, T, D)
        h_tan = self.final_norm(h_tan)
        scores = self.scorer(h_tan).squeeze(-1)     # (B, T)

        return scores, h