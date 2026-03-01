# coding=utf-8
"""
Temporal ordering models:
  1) TemporalViT            : ViT → TemporalHead → Plackett-Luce loss  (Euclidean)
  2) HyperbolicTemporalViT  : ViT → Lorentz projection → Entailment loss (Hyperbolic)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt

from models.modeling import VisionTransformer, CONFIGS


# ═════════════════════════════════════════════
# Lorentz helper (only one — not in geoopt)
# ═════════════════════════════════════════════

def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product between points: ⟨x,y⟩_L = −x₀y₀ + Σ xᵢyᵢ.
    geoopt.inner is for tangent vectors; this operates on manifold points."""
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)


# ═════════════════════════════════════════════
# Euclidean (기존 코드 그대로)
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
# Hyperbolic  (Lorentz model via geoopt)
# ═════════════════════════════════════════════

class LorentzProjection(nn.Module):
    """
    Euclidean  →  linear proj  →  tangent vector at origin  →  expmap  →  hyperboloid
    """

    def __init__(self, in_dim: int, out_dim: int, curvature: float = 1.0):
        super().__init__()
        self.manifold = geoopt.Lorentz(k=curvature)
        self.proj = nn.Linear(in_dim, out_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.proj(x) * self.scale                             # [..., out_dim]
        t = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # [..., out_dim+1]
        origin = self.manifold.origin(t.shape, device=v.device, dtype=v.dtype)
        return self.manifold.expmap(origin, t)


class HyperbolicEntailmentLoss(nn.Module):
    """
    Entailment-cone loss on Lorentz hyperboloid.

    For every ordered pair (i < j):  frame i  entails  frame j.

    Half-aperture  (eq 10):
        aper(x) = arcsin( 2K / (√c · ‖x_space‖) )

    Exterior angle (eq 11):
        ext(x, y) = arccos( (y_time + x_time · c · ⟨x,y⟩_L)
                             / (‖x_space‖ · √((c · ⟨x,y⟩_L)² − 1)) )

    Entailment condition:  ext(x, y) ≥ aper(x)   →  y inside x's cone
    Cone loss:  max(0,  aper(x) − ext(x, y) + margin)

    Height ordering (supplementary):
        d(o, xᵢ) + margin < d(o, xⱼ)   for i < j
    """

    def __init__(self, curvature: float = 1.0,
                 K: float = 0.1,
                 height_margin: float = 0.1,
                 cone_margin: float = 0.05,
                 height_weight: float = 1.0,
                 cone_weight: float = 1.0):
        super().__init__()
        self.manifold = geoopt.Lorentz(k=curvature)
        self.c = curvature
        self.K = K                    # aperture constant (paper: K=0.1)
        self.height_margin = height_margin
        self.cone_margin = cone_margin
        self.height_weight = height_weight
        self.cone_weight = cone_weight

    def _half_aperture(self, h: torch.Tensor) -> torch.Tensor:
        """Eq (10):  aper(x) = arcsin( 2K / (√c · ‖x_space‖) )"""
        x_space_norm = h[..., 1:].norm(dim=-1)               # ‖x_space‖
        sin_val = (2 * self.K) / (math.sqrt(self.c) * x_space_norm + 1e-6)
        sin_val = torch.clamp(sin_val, max=1.0 - 1e-6)
        return torch.asin(sin_val)

    def _exterior_angle(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Eq (11):  ext(x, y) = arccos( numerator / denominator )

        numerator   = y_time + x_time · c · ⟨x,y⟩_L
        denominator = ‖x_space‖ · √( (c · ⟨x,y⟩_L)² − 1 )
        """
        x_time = x[..., 0]                                   # x₀
        y_time = y[..., 0]                                    # y₀
        x_space_norm = x[..., 1:].norm(dim=-1)                # ‖x_space‖

        c_inner = self.c * lorentz_inner(x, y)                # c · ⟨x,y⟩_L

        numer = y_time + x_time * c_inner
        denom = x_space_norm * torch.sqrt(torch.clamp(c_inner ** 2 - 1, min=1e-10))

        cos_val = numer / (denom + 1e-8)
        cos_val = torch.clamp(cos_val, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cos_val)

    def forward(self, h: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        h : [B, T, D+1]  points on hyperboloid (chronological order)
        """
        B, T, D1 = h.shape
        device = h.device

        # ── ordered pair indices (i < j) ──
        idx_i, idx_j = [], []
        for i in range(T):
            for j in range(i + 1, T):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device)
        idx_j = torch.tensor(idx_j, device=device)
        P = len(idx_i)

        h_i = h[:, idx_i]                                    # [B, P, D+1]
        h_j = h[:, idx_j]

        # ── 1) Height ordering loss ──
        origin = self.manifold.origin(h_i.shape, device=device, dtype=h.dtype)
        d_i = self.manifold.dist(origin, h_i)                # [B, P]
        d_j = self.manifold.dist(origin, h_j)
        height_loss = F.relu(d_i - d_j + self.height_margin).mean()

        # ── 2) Cone inclusion loss ──
        aperture = self._half_aperture(h_i)                   # [B, P]
        ext = self._exterior_angle(h_i, h_j)                  # [B, P]

        # y inside cone  ⟺  ext(x,y) ≥ aper(x)
        # penalize violation:
        cone_loss = F.relu(aperture - ext + self.cone_margin).mean()

        return self.height_weight * height_loss + self.cone_weight * cone_loss


# ── evaluation metrics ──────────────────────

@torch.no_grad()
def hyperbolic_ordering_accuracy(h: torch.Tensor,
                                 manifold: geoopt.Lorentz = None) -> float:
    """Fraction of pairs where d(o, xᵢ) < d(o, xⱼ) for i < j."""
    if manifold is None:
        manifold = geoopt.Lorentz(k=1.0)
    B, T, D1 = h.shape
    origin = manifold.origin((B, T, D1), device=h.device, dtype=h.dtype)
    d = manifold.dist(origin, h)                              # [B, T]

    correct, total = 0, 0
    for i in range(T):
        for j in range(i + 1, T):
            correct += (d[:, i] < d[:, j]).sum().item()
            total += B
    return correct / max(total, 1)


@torch.no_grad()
def hyperbolic_cone_accuracy(h: torch.Tensor, K: float = 0.1,
                             curvature: float = 1.0) -> float:
    """Fraction of pairs where ext(xᵢ, xⱼ) ≥ aper(xᵢ)  (y inside cone)."""
    B, T, _ = h.shape
    sqrt_c = math.sqrt(curvature)

    correct, total = 0, 0
    for i in range(T):
        for j in range(i + 1, T):
            x, y = h[:, i], h[:, j]

            # aperture
            x_space_norm = x[..., 1:].norm(dim=-1)
            sin_val = torch.clamp((2 * K) / (sqrt_c * x_space_norm + 1e-6),
                                  max=1.0 - 1e-6)
            aperture = torch.asin(sin_val)

            # exterior angle
            c_inner = curvature * lorentz_inner(x, y)
            numer = y[..., 0] + x[..., 0] * c_inner
            denom = x_space_norm * torch.sqrt(torch.clamp(c_inner ** 2 - 1, min=1e-10))
            cos_val = torch.clamp(numer / (denom + 1e-8), -1.0 + 1e-6, 1.0 - 1e-6)
            ext = torch.acos(cos_val)

            correct += (ext >= aperture).sum().item()
            total += B
    return correct / max(total, 1)


# ═════════════════════════════════════════════
# Hyperbolic Full Model
# ═════════════════════════════════════════════

class HyperbolicTemporalViT(nn.Module):
    """
    ViT backbone → CLS token → Lorentz projection → hyperboloid points
    """

    def __init__(self, config, img_size=224, pretrained_weights=None,
                 hyp_dim=128, curvature=1.0, zero_head=True, vis=False):
        super().__init__()
        self.backbone = VisionTransformer(
            config, img_size=img_size, num_classes=1,
            zero_head=zero_head, vis=vis,
        )
        self.lorentz_proj = LorentzProjection(
            in_dim=config.hidden_size, out_dim=hyp_dim, curvature=curvature,
        )
        if pretrained_weights is not None:
            self.backbone.load_from(pretrained_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, T, 3, H, W]  →  [B, T, hyp_dim+1] on hyperboloid"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        encoded, _ = self.backbone.transformer(x)
        cls = encoded[:, 0].view(B, T, -1)
        return self.lorentz_proj(cls)