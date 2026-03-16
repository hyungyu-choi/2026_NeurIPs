# coding=utf-8
"""
Angular Contrastive Head for Lorentz Hyperbolic Embeddings.

Design principle:
    On the Lorentz hyperboloid, each embedding h_space can be decomposed into:
        - radial component:  r = ||h_space||   (encodes temporal position)
        - angular component: â = h_space / ||h_space||  (encodes visual content)

    The existing entailment + PL losses constrain the radial axis.
    This module constrains the angular axis via contrastive learning:
        "Two augmentations of the same frame should have similar angular components,
         while different frames should have different angular components."

    Since ∂L_radial/∂h and ∂L_angular/∂h are orthogonal in the Riemannian sense,
    the two losses do not interfere with each other during optimization.

Components:
    1. AngularDecomposition  : h_space → (r, â)
    2. AngularProjectionHead : â → projected features for contrastive loss
    3. AngularContrastiveLoss: InfoNCE on projected angular features
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularDecomposition(nn.Module):
    """
    Decompose Lorentz hyperboloid space components into radial and angular parts.

    Given h_space ∈ ℝ^D (space components on the hyperboloid):
        radial:  r = ||h_space||₂
        angular: â = h_space / (||h_space||₂ + ε)

    This is the polar coordinate decomposition on the hyperboloid.
    The angular component â lives on the unit sphere S^{D-1}.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, h_space: torch.Tensor):
        """
        Args:
            h_space: (..., D) space components on the hyperboloid
        Returns:
            radial:  (...,) scalar norms
            angular: (..., D) unit vectors on S^{D-1}
        """
        radial = torch.norm(h_space, dim=-1)                          # (...,)
        angular = h_space / (radial.unsqueeze(-1) + self.eps)          # (..., D)
        return radial, angular


class AngularProjectionHead(nn.Module):
    """
    MLP projection head for angular contrastive learning.

    Following SimCLR/MoCo convention, the contrastive loss is applied on
    the output of this projection head (not the angular representation itself).
    At evaluation time, this head is discarded and the angular representation
    â is used directly.

    Architecture:
        â (D) → Linear(D, D_proj) → BatchNorm → ReLU → Linear(D_proj, D_proj) → L2-norm
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, angular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angular: (N, D) unit vectors (angular components)
        Returns:
            (N, D_proj) L2-normalized projected features
        """
        z = self.net(angular)
        z = F.normalize(z, dim=-1)
        return z


class AngularContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss on angular components.

    For each positive pair (same frame, different augmentation):
        L = -log [ exp(sim(z_a, z_b) / τ) / Σ_{negatives} exp(sim(z_a, z_neg) / τ) ]

    Positive pairs: (z_temporal[b,k], z_contrast[b,k]) for matching frames
    Negatives: all other z_contrast in the batch

    This loss encourages:
        - Same frame's angular components to be similar regardless of augmentation
        - Different frames' angular components to be dissimilar
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_temporal: torch.Tensor,
        z_contrast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_temporal: (N, D_proj) projected angular features from augmentation A
            z_contrast: (N, D_proj) projected angular features from augmentation B
                        N = B * K (batch_size * contrast_k)
                        z_temporal[i] and z_contrast[i] are a positive pair.
        Returns:
            scalar InfoNCE loss
        """
        N = z_temporal.shape[0]
        device = z_temporal.device

        # Both should already be L2-normalized, but ensure it
        z_a = F.normalize(z_temporal, dim=-1)
        z_b = F.normalize(z_contrast, dim=-1)

        # Similarity matrix: (N, N)
        # sim[i, j] = cosine_similarity(z_a[i], z_b[j]) / τ
        sim = torch.mm(z_a, z_b.t()) / self.temperature    # (N, N)

        # Positive pairs are on the diagonal: sim[i, i]
        # InfoNCE: for each row i, the target is column i
        labels = torch.arange(N, device=device)

        # Symmetric loss: both directions
        loss_ab = F.cross_entropy(sim, labels)
        loss_ba = F.cross_entropy(sim.t(), labels)

        loss = (loss_ab + loss_ba) / 2.0

        # Compute accuracy for logging
        with torch.no_grad():
            pred_ab = sim.argmax(dim=1)
            pred_ba = sim.t().argmax(dim=1)
            acc = ((pred_ab == labels).float().mean() +
                   (pred_ba == labels).float().mean()) / 2.0

        return loss, acc