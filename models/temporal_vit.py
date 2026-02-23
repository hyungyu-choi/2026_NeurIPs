# coding=utf-8
"""
Temporal ordering model:
  ViT backbone  →  TemporalHead (reduce + TransformerEncoder + scorer)  →  Plackett-Luce loss
"""
import torch
import torch.nn as nn
from models.modeling import VisionTransformer, CONFIGS


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
class PlackettLuceLoss(nn.Module):
    """
    Self-supervised temporal ordering:
        logits : [B, T]  – higher = earlier in time
        labels : [B, T] or None; if None → chronological 0..T-1
    """
    def __init__(self, sample=False, R=4, K=8):
        super().__init__()
        self.sample, self.R, self.K = sample, R, K

    def forward(self, logits, labels=None):
        s = logits.float() # [32, 8]
        B, T = s.shape
        dev = s.device

        if labels is None:
            labels = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1) # [32, [0, 1, 2, 3, 4, 5, 6, 7]]

        def list_nll(score_vec):
            lse = torch.logcumsumexp(score_vec.flip(0), 0).flip(0)
            return (lse - score_vec).sum()

        total, count = 0.0, 0
        for b in range(B):
            sb, lb = s[b], labels[b]
            if not self.sample:
                perm = torch.argsort(lb) # [0, 1, 2, 3, 4, 5, 6, 7]
                total += list_nll(sb[perm])
                count += T
            else:
                for _ in range(self.R):
                    idx = torch.randperm(T, device=dev)[:self.K]
                    order = torch.argsort(lb[idx])
                    total += list_nll(sb[idx[order]])
                    count += self.K
        return total / count


# ─────────────────────────────────────────────
# Temporal Head
# ─────────────────────────────────────────────
class TemporalSideContext(nn.Module):
    def __init__(self, D, max_len=64, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            D, n_head, 4 * D,
            dropout=dropout, batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):          # x [B, T, D]
        return self.enc(x)         # [B, T, D]


class TemporalHead(nn.Module):
    """
    Converts backbone features [B, T, D] → logits [B, T, 1] for Plackett-Luce.
    """
    def __init__(self, backbone_dim: int, hidden_mul: float = 0.5,
                 max_len: int = 64, n_layers: int = 6, n_head: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(backbone_dim * hidden_mul)

        self.reduce = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = TemporalSideContext(
            hidden_dim, max_len=max_len,
            n_layers=n_layers, n_head=n_head, dropout=dropout,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):     # x : [B, T, D]
        x = self.reduce(x)                  # [B, T, hidden]
        x = self.temporal(x)                # [B, T, hidden]
        return self.scorer(x)               # [B, T, 1]


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────
class TemporalViT(nn.Module):
    """
    ViT backbone  →  TemporalHead  →  [B, T] logits

    Input:  [B, T, 3, H, W]
    Output: [B, T]
    """
    def __init__(self, config, img_size=224, pretrained_weights=None,
                 hidden_mul=0.5, max_len=64, n_layers=6, n_head=8,
                 dropout=0.1, zero_head=True, vis=False):
        super().__init__()
        self.backbone = VisionTransformer(
            config, img_size=img_size, num_classes=1,
            zero_head=zero_head, vis=vis,
        )
        hidden = config.hidden_size

        self.temporal_head = TemporalHead(
            backbone_dim=hidden,
            hidden_mul=hidden_mul,
            max_len=max_len,
            n_layers=n_layers,
            n_head=n_head,
            dropout=dropout,
        )

        if pretrained_weights is not None:
            self.backbone.load_from(pretrained_weights)

    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        returns: [B, T]
        """
        B, T, C, H, W = x.shape

        # Merge batch & time → [B*T, 3, H, W]
        x = x.view(B * T, C, H, W)

        # ViT encoder → cls token features
        encoded, _ = self.backbone.transformer(x)   # [B*T, n_patches+1, hidden]
        cls_features = encoded[:, 0]                 # [B*T, hidden]

        # Reshape back → [B, T, hidden]
        cls_features = cls_features.view(B, T, -1)

        # Temporal head → [B, T, 1] → [B, T]
        logits = self.temporal_head(cls_features).squeeze(-1)

        return logits