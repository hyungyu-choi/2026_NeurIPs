# coding=utf-8
"""
Temporal ordering models:
  1) TemporalViT            : ViT → TemporalHead → Plackett-Luce loss  (Euclidean)
  2) HyperbolicTemporalViT  : ViT → Lorentz exp_map → Entailment loss  (Hyperbolic, MERU-style)

The hyperbolic model follows the MERU architecture:
  - Only space components are stored/passed (time is computed on-the-fly)
  - Learnable curvature parameter (stored as log)
  - Learnable scaling factor (alpha) before exp_map
  - Entailment loss: clamp(oxy_angle - half_aperture, min=0)
  - Height ordering via geodesic distance to origin
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modeling import VisionTransformer, CONFIGS
from models import lorentz_ops as L


# ═════════════════════════════════════════════
# Euclidean (unchanged)
# ═════════════════════════════════════════════

class PlackettLuceLoss(nn.Module):
    def __init__(self, sample=False, R=4, K=8):
        super().__init__()
        self.sample, self.R, self.K = sample, R, K

    def forward(self, logits, labels=None):
        s = logits.float()
        B, T = s.shape
        dev = s.device

        if labels is None:
            labels = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)

        def list_nll(score_vec):
            lse = torch.logcumsumexp(score_vec.flip(0), 0).flip(0)
            return (lse - score_vec).sum()

        total, count = 0.0, 0
        for b in range(B):
            sb, lb = s[b], labels[b]
            if not self.sample:
                perm = torch.argsort(lb)
                total += list_nll(sb[perm])
                count += T
            else:
                for _ in range(self.R):
                    idx = torch.randperm(T, device=dev)[:self.K]
                    order = torch.argsort(lb[idx])
                    total += list_nll(sb[idx[order]])
                    count += self.K
        return total / count


class TemporalSideContext(nn.Module):
    def __init__(self, D, max_len=64, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            D, n_head, 4 * D, dropout=dropout, batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.enc(x)


class TemporalHead(nn.Module):
    def __init__(self, backbone_dim, hidden_mul=0.5,
                 max_len=64, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        hidden_dim = int(backbone_dim * hidden_mul)
        self.reduce = nn.Sequential(nn.Linear(backbone_dim, hidden_dim), nn.GELU())
        self.temporal = TemporalSideContext(
            hidden_dim, max_len=max_len,
            n_layers=n_layers, n_head=n_head, dropout=dropout,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.scorer(self.temporal(self.reduce(x)))


class TemporalViT(nn.Module):
    def __init__(self, config, img_size=224, pretrained_weights=None,
                 hidden_mul=0.5, max_len=64, n_layers=6, n_head=8,
                 dropout=0.1, zero_head=True, vis=False):
        super().__init__()
        self.backbone = VisionTransformer(
            config, img_size=img_size, num_classes=1,
            zero_head=zero_head, vis=vis,
        )
        self.temporal_head = TemporalHead(
            backbone_dim=config.hidden_size, hidden_mul=hidden_mul,
            max_len=max_len, n_layers=n_layers,
            n_head=n_head, dropout=dropout,
        )
        if pretrained_weights is not None:
            self.backbone.load_from(pretrained_weights)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        encoded, _ = self.backbone.transformer(x)
        cls = encoded[:, 0].view(B, T, -1)
        return self.temporal_head(cls).squeeze(-1)


# ═════════════════════════════════════════════
# Hyperbolic (MERU-style, no geoopt)
# ═════════════════════════════════════════════

class MERUStyleProjection(nn.Module):
    """
    MERU-style projection: Euclidean CLS → linear → scale → exp_map0 → hyperboloid.

    Following MERU:
      - Linear projection to embed_dim
      - Learnable alpha scaling (clamped to not upscale)
      - Exponential map at origin to lift onto hyperboloid
      - Only space components are output (time computed on-the-fly)
    """

    def __init__(self, in_dim: int, embed_dim: int, curv_init: float = 1.0,
                 learn_curv: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

        # Learnable curvature stored as log (hyperboloid curvature = -exp(curv))
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scaling factor (initialized so features have ~unit norm)
        self.alpha = nn.Parameter(torch.tensor(embed_dim ** -0.5).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) Euclidean features
        Returns:
            (..., embed_dim) space components on the hyperboloid
        """
        # Clamp curvature
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        # Clamp alpha so it doesn't upscale
        self.alpha.data = torch.clamp(self.alpha.data, max=0.0)

        v = self.proj(x)                          # (..., embed_dim)
        v = v * self.alpha.exp()                   # scale

        # Lift to hyperboloid via exp_map at origin (float32 for stability)
        if x.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float32):
                h = L.exp_map0(v.float(), self.curv.exp())
        else:
            h = L.exp_map0(v.float(), self.curv.exp())

        return h

    @property
    def curvature(self) -> torch.Tensor:
        """Current curvature value (positive scalar)."""
        return self.curv.exp()


class HyperbolicEntailmentLoss(nn.Module):
    """
    MERU-style entailment loss for temporal ordering.

    For each ordered pair (i < j), frame i should entail frame j:
      - Entailment cone loss: clamp(oxy_angle(i,j) - half_aperture(i), min=0)
      - Height ordering loss: clamp(dist(o,i) - dist(o,j) + margin, min=0)

    All operations use only space components (MERU convention).
    The curvature parameter is shared from the projection module.
    """

    def __init__(
        self,
        min_radius: float = 0.1,
        height_margin: float = 0.1,
        height_weight: float = 1.0,
        cone_weight: float = 1.0,
    ):
        super().__init__()
        self.min_radius = min_radius
        self.height_margin = height_margin
        self.height_weight = height_weight
        self.cone_weight = cone_weight

    def forward(self, h: torch.Tensor, curv: torch.Tensor) -> dict:
        """
        Args:
            h: (B, T, D) space components on hyperboloid (chronological order)
            curv: positive scalar curvature (from projection module)
        Returns:
            dict with 'loss', 'entailment_loss', 'height_loss'
        """
        B, T, D = h.shape
        device = h.device

        # Build ordered pair indices (i < j)
        idx_i, idx_j = [], []
        for i in range(T):
            for j in range(i + 1, T):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device)
        idx_j = torch.tensor(idx_j, device=device)

        # (B, P, D) where P = T*(T-1)/2
        h_i = h[:, idx_i]
        h_j = h[:, idx_j]

        # Force float32 for numerical stability
        B_P = B * len(idx_i)
        hi_flat = h_i.reshape(B_P, D).float()
        hj_flat = h_j.reshape(B_P, D).float()
        _curv = curv.float() if isinstance(curv, torch.Tensor) else curv

        # ── 1) Entailment cone loss (MERU-style) ──
        _angle = L.oxy_angle(hi_flat, hj_flat, _curv)
        _aperture = L.half_aperture(hi_flat, _curv, min_radius=self.min_radius)
        entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

        # ── 2) Height ordering loss ──
        d_i = L.dist_to_origin(hi_flat, _curv)
        d_j = L.dist_to_origin(hj_flat, _curv)
        height_loss = torch.clamp(
            d_i - d_j + self.height_margin, min=0
        ).mean()

        loss = (self.cone_weight * entailment_loss
                + self.height_weight * height_loss)

        return {
            "loss": loss,
            "entailment_loss": entailment_loss,
            "height_loss": height_loss,
        }


# ── Evaluation metrics ──────────────────────

@torch.no_grad()
def hyperbolic_ordering_accuracy(
    h: torch.Tensor, curv: float | torch.Tensor = 1.0
) -> float:
    """Fraction of pairs where dist(o, x_i) < dist(o, x_j) for i < j."""
    B, T, D = h.shape
    # (B, T)
    h_flat = h.reshape(B * T, D)
    d_flat = L.dist_to_origin(h_flat, curv)
    d = d_flat.reshape(B, T)

    correct, total = 0, 0
    for i in range(T):
        for j in range(i + 1, T):
            correct += (d[:, i] < d[:, j]).sum().item()
            total += B
    return correct / max(total, 1)


@torch.no_grad()
def hyperbolic_cone_accuracy(
    h: torch.Tensor, curv: float | torch.Tensor = 1.0,
    min_radius: float = 0.1
) -> float:
    """Fraction of pairs where frame_i's cone contains frame_j."""
    B, T, D = h.shape

    correct, total = 0, 0
    for i in range(T):
        for j in range(i + 1, T):
            x = h[:, i]  # (B, D)
            y = h[:, j]  # (B, D)

            aperture = L.half_aperture(x, curv, min_radius=min_radius)
            ext = L.oxy_angle(x, y, curv)

            correct += (ext <= aperture).sum().item()
            total += B
    return correct / max(total, 1)


# ═════════════════════════════════════════════
# Hyperbolic Full Model (MERU-style)
# ═════════════════════════════════════════════

class HyperbolicTemporalViT(nn.Module):
    """
    ViT backbone → CLS token → MERU-style Lorentz projection → space components

    Output is (B, T, embed_dim) of space components on the hyperboloid.
    """

    def __init__(self, config, img_size=224, pretrained_weights=None,
                 embed_dim: int = 128, curv_init: float = 1.0,
                 learn_curv: bool = True, zero_head=True, vis=False):
        super().__init__()
        self.backbone = VisionTransformer(
            config, img_size=img_size, num_classes=1,
            zero_head=zero_head, vis=vis,
        )
        self.lorentz_proj = MERUStyleProjection(
            in_dim=config.hidden_size,
            embed_dim=embed_dim,
            curv_init=curv_init,
            learn_curv=learn_curv,
        )
        if pretrained_weights is not None:
            self.backbone.load_from(pretrained_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3, H, W) → (B, T, embed_dim) space components on hyperboloid
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        encoded, _ = self.backbone.transformer(x)
        cls = encoded[:, 0].view(B, T, -1)       # (B, T, hidden_size)
        return self.lorentz_proj(cls)              # (B, T, embed_dim)

    @property
    def curvature(self) -> torch.Tensor:
        return self.lorentz_proj.curvature