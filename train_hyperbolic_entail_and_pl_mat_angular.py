# coding=utf-8
"""
Train ViT with combined MERU-style hyperbolic entailment loss + Plackett-Luce
temporal ordering loss at MULTIPLE SCALES (MAT) + Angular Contrastive SSL.

This extends train_hyperbolic_entail_and_pl_mat.py with an additional
angular contrastive loss that teaches the model visual content information.

Key insight (Radial-Angular Decomposition):
    On the Lorentz hyperboloid, each embedding h_space decomposes into:
        - radial:  r = ||h_space||   → constrained by entailment + PL losses
        - angular: â = h_space / ||h||→ constrained by angular contrastive loss

    The gradients of radial losses and angular losses are orthogonal at h_space,
    so they do not interfere with each other during optimization.

Architecture:
    Input: x_temporal (B,T,3,H,W) + x_contrast (B,K,3,H,W)
        │
        ▼  Shared ViT + MERUStyleProjection
        │
        ├── h_temporal (B,T,D) ──→ [Radial Branch: existing MAT pipeline]
        │                            Pre-split LorentzBlocks → Multi-scale split
        │                            → Per-scale LorentzScoreHead
        │                            → Entailment + PL losses
        │
        └── h_contrast (B,K,D) ──→ [Angular Branch: new]
             + h_temporal[:,selected]    Angular decomposition (â = h/||h||)
                                         → AngularProjectionHead
                                         → InfoNCE contrastive loss

    Total loss = Σ_s w_s · L_ordering_s  +  λ_ang · L_angular
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import math
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.modeling import CONFIGS
from models.temporal_vit import (
    HyperbolicTemporalViT,
    HyperbolicEntailmentLoss,
    PlackettLuceLoss,
    hyperbolic_ordering_accuracy,
    hyperbolic_cone_accuracy,
)
from models.lorentz_head import LorentzScoreHead, LorentzBlock
from models.angular_head import (
    AngularDecomposition,
    AngularProjectionHead,
    AngularContrastiveLoss,
)
from models import lorentz_ops as L
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_angular import get_angular_loader

# Re-use dimension reduction and scale weights from the MAT module
from train_hyperbolic_entail_and_pl_mat import (
    HyperbolicDimReduction,
    LearnableScaleWeights,
)

logger = logging.getLogger(__name__)


# =============================================
# Model: MAT + Angular Contrastive
# =============================================

class MultiScaleAngularModel(nn.Module):
    """
    Multi-Scale Hyperbolic MAT model with Angular Contrastive branch.

    Extends MultiScaleHyperbolicCombinedModel with:
      - AngularDecomposition: extracts angular components from h_space
      - AngularProjectionHead: projects angular components for contrastive loss
    """

    def __init__(
        self,
        config,
        img_size: int = 224,
        pretrained_weights=None,
        embed_dim: int = 128,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        # Pre-split Lorentz interaction
        pre_split_n_layers: int = 2,
        pre_split_n_heads: int = 4,
        pre_split_mlp_ratio: float = 4.0,
        pre_split_dropout: float = 0.1,
        # Per-scale score head
        score_n_layers: int = 2,
        score_n_heads: int = 4,
        score_mlp_ratio: float = 4.0,
        score_dropout: float = 0.1,
        # Scale weight
        scale_weight_temp: float = 1.0,
        scale_min_weight: float = 0.01,
        # Angular contrastive
        angular_proj_dim: int = 128,
        zero_head: bool = True,
        vis: bool = False,
    ):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

        dim_half = embed_dim // 2
        dim_quarter = embed_dim // 4

        # ── Encoder: ViT → hyperboloid (dim D) ──
        self.encoder = HyperbolicTemporalViT(
            config,
            img_size=img_size,
            pretrained_weights=pretrained_weights,
            embed_dim=embed_dim,
            curv_init=curv_init,
            learn_curv=learn_curv,
            zero_head=zero_head,
            vis=vis,
        )

        # ── Pre-split Lorentz interaction blocks (full dim D) ──
        def _safe_heads(dim, desired_heads):
            h = desired_heads
            while dim % h != 0 and h > 1:
                h -= 1
            return h

        self.pre_split_blocks = nn.ModuleList([
            LorentzBlock(
                embed_dim,
                _safe_heads(embed_dim, pre_split_n_heads),
                pre_split_mlp_ratio,
                pre_split_dropout,
            )
            for _ in range(pre_split_n_layers)
        ])

        # ── Hyperbolic dimension reductions ──
        self.reduce_half = HyperbolicDimReduction(embed_dim, dim_half)
        self.reduce_quarter = HyperbolicDimReduction(embed_dim, dim_quarter)

        # ── Per-scale Lorentz Score Heads ──
        self.score_head_full = LorentzScoreHead(
            embed_dim=embed_dim,
            n_layers=score_n_layers,
            n_heads=_safe_heads(embed_dim, score_n_heads),
            mlp_ratio=score_mlp_ratio,
            dropout=score_dropout,
        )
        self.score_head_half = LorentzScoreHead(
            embed_dim=dim_half,
            n_layers=score_n_layers,
            n_heads=_safe_heads(dim_half, score_n_heads),
            mlp_ratio=score_mlp_ratio,
            dropout=score_dropout,
        )
        self.score_head_quarter = LorentzScoreHead(
            embed_dim=dim_quarter,
            n_layers=score_n_layers,
            n_heads=_safe_heads(dim_quarter, score_n_heads),
            mlp_ratio=score_mlp_ratio,
            dropout=score_dropout,
        )

        # ── Learnable scale weights ──
        self.scale_weights = LearnableScaleWeights(
            n_scales=3, init_temp=scale_weight_temp,
            min_weight=scale_min_weight,
        )

        # ── Angular Contrastive branch ──
        self.angular_decomp = AngularDecomposition()
        self.angular_proj = AngularProjectionHead(
            in_dim=embed_dim,
            proj_dim=angular_proj_dim,
        )

        self.embed_dim = embed_dim
        self.dim_half = dim_half
        self.dim_quarter = dim_quarter

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw frames to hyperboloid embeddings.

        Args:
            x: (N, 3, H, W) individual frames (NOT batched temporal clips)
        Returns:
            (N, D) space components on hyperboloid
        """
        encoded, _ = self.encoder.backbone.transformer(x)
        cls = encoded[:, 0]                           # (N, hidden_size)
        h = self.encoder.lorentz_proj(cls)             # (N, D) on hyperboloid
        return h

    def forward_temporal(self, x_temporal: torch.Tensor):
        """
        Forward pass for the temporal ordering (radial) branch.

        Args:
            x_temporal: (B, T, 3, H, W)
        Returns:
            dict with multi-scale outputs (same as original MAT)
            h_proj_raw: (B, T, D) embeddings BEFORE pre-split blocks
                        (used for angular decomposition)
        """
        curv = self.encoder.curvature

        # Encode to hyperboloid
        h = self.encoder(x_temporal)    # (B, T, D)
        h_proj_raw = h                  # Save for angular branch

        # Pre-split Lorentz interaction
        for block in self.pre_split_blocks:
            h = block(h, curv)

        h_full = h

        # Multi-scale split
        h_half = self.reduce_half(h_full, curv)
        h_quarter = self.reduce_quarter(h_full, curv)

        # Score heads
        scores_full, h_ref_full = self.score_head_full(h_full, curv)
        scores_half, h_ref_half = self.score_head_half(h_half, curv)
        scores_quarter, h_ref_quarter = self.score_head_quarter(h_quarter, curv)

        sw = self.scale_weights()

        return {
            'full': {'h_proj': h_full, 'scores': scores_full, 'h_ref': h_ref_full},
            'half': {'h_proj': h_half, 'scores': scores_half, 'h_ref': h_ref_half},
            'quarter': {'h_proj': h_quarter, 'scores': scores_quarter, 'h_ref': h_ref_quarter},
            'scale_weights': sw,
            'h_proj_raw': h_proj_raw,
        }

    def forward_angular(
        self,
        h_temporal_selected: torch.Tensor,
        h_contrast: torch.Tensor,
    ):
        """
        Forward pass for the angular contrastive branch.

        Args:
            h_temporal_selected: (B*K, D) hyperboloid embeddings of selected frames
                                 (from augmentation A)
            h_contrast:          (B*K, D) hyperboloid embeddings of same frames
                                 (from augmentation B)
        Returns:
            z_temporal: (B*K, proj_dim) projected angular features
            z_contrast: (B*K, proj_dim) projected angular features
            angular_stats: dict with norms and angular cosine sim for logging
        """
        # Angular decomposition
        r_temporal, a_temporal = self.angular_decomp(h_temporal_selected)
        r_contrast, a_contrast = self.angular_decomp(h_contrast)

        # Projection head
        z_temporal = self.angular_proj(a_temporal)
        z_contrast = self.angular_proj(a_contrast)

        # Stats for logging
        with torch.no_grad():
            # Cosine similarity between paired angular components
            cos_sim = F.cosine_similarity(a_temporal, a_contrast, dim=-1).mean()

        angular_stats = {
            'radial_mean_temporal': r_temporal.mean().item(),
            'radial_mean_contrast': r_contrast.mean().item(),
            'angular_cosine_sim': cos_sim.item(),
        }

        return z_temporal, z_contrast, angular_stats

    def forward(self, x_temporal, x_contrast=None, contrast_indices=None):
        """
        Full forward pass.

        Args:
            x_temporal:       (B, T, 3, H, W)
            x_contrast:       (B, K, 3, H, W) or None
            contrast_indices: (B, K) or None
        Returns:
            temporal_out: dict with multi-scale outputs
            angular_out:  dict with z_temporal, z_contrast, stats (or None)
        """
        # ── Temporal (radial) branch ──
        temporal_out = self.forward_temporal(x_temporal)

        # ── Angular branch ──
        angular_out = None
        if x_contrast is not None and contrast_indices is not None:
            B, K = x_contrast.shape[:2]
            D = self.embed_dim

            # Encode contrastive views through shared backbone
            x_contrast_flat = x_contrast.view(B * K, *x_contrast.shape[2:])
            h_contrast_flat = self.encode_frames(x_contrast_flat)   # (B*K, D)

            # Select corresponding temporal embeddings using indices
            # h_proj_raw is (B, T, D) — BEFORE pre-split blocks
            h_raw = temporal_out['h_proj_raw']  # (B, T, D)
            h_temporal_selected = torch.zeros(B, K, D, device=h_raw.device,
                                               dtype=h_raw.dtype)
            for b in range(B):
                h_temporal_selected[b] = h_raw[b, contrast_indices[b]]
            h_temporal_flat = h_temporal_selected.view(B * K, D)

            # Forward angular branch
            z_temporal, z_contrast, angular_stats = self.forward_angular(
                h_temporal_flat, h_contrast_flat,
            )
            angular_out = {
                'z_temporal': z_temporal,
                'z_contrast': z_contrast,
                'stats': angular_stats,
            }

        return temporal_out, angular_out

    @property
    def curvature(self) -> torch.Tensor:
        return self.encoder.curvature


# =============================================
# Helpers
# =============================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(args, model, step=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    suffix = f"_step{step}" if step else ""
    save_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{args.name}{suffix}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), path)
    logger.info(f"Saved model checkpoint to {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def kendall_tau_accuracy(scores: torch.Tensor) -> float:
    B, T = scores.shape
    pred_order = torch.argsort(scores, dim=1, descending=True)
    correct, total = 0, 0
    for b in range(B):
        po = pred_order[b]
        for i in range(T):
            for j in range(i + 1, T):
                if po[i] < po[j]:
                    correct += 1
                total += 1
    return correct / max(total, 1)


# =============================================
# Setup
# =============================================

def setup(args):
    config = CONFIGS[args.model_type]

    pretrained_weights = None
    if args.pretrained_dir is not None:
        pretrained_weights = np.load(args.pretrained_dir)
        logger.info(f"Loaded pretrained weights from {args.pretrained_dir}")
    else:
        logger.info("Training from scratch (no pretrained weights)")

    model = MultiScaleAngularModel(
        config,
        img_size=args.img_size,
        pretrained_weights=pretrained_weights,
        embed_dim=args.embed_dim,
        curv_init=args.curv_init,
        learn_curv=args.learn_curv,
        pre_split_n_layers=args.pre_split_n_layers,
        pre_split_n_heads=args.pre_split_n_heads,
        pre_split_mlp_ratio=args.pre_split_mlp_ratio,
        pre_split_dropout=args.pre_split_dropout,
        score_n_layers=args.score_n_layers,
        score_n_heads=args.score_n_heads,
        score_mlp_ratio=args.score_mlp_ratio,
        score_dropout=args.score_dropout,
        scale_weight_temp=args.scale_weight_temp,
        scale_min_weight=args.scale_min_weight,
        angular_proj_dim=args.angular_proj_dim,
        zero_head=True,
    )
    model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Config: {config}")
    logger.info(f"Training parameters: {args}")
    logger.info(f"Total trainable parameters: {num_params:.1f}M")
    logger.info(f"Multi-scale dims: full={args.embed_dim}, "
                f"half={args.embed_dim // 2}, quarter={args.embed_dim // 4}")
    logger.info(f"Angular contrastive: K={args.contrast_k}, "
                f"proj_dim={args.angular_proj_dim}, "
                f"λ={args.angular_weight}, τ={args.angular_temperature}")
    return args, model


# =============================================
# Validation
# =============================================

@torch.no_grad()
def valid(args, model, entailment_criterions, pl_criterion, writer,
          test_loader, global_step):
    """
    Validation uses only the temporal ordering metrics (no angular contrastive
    during validation since the val loader returns single-augmented clips).
    """
    model.eval()

    scale_names = ['full', 'half', 'quarter']
    meters = {s: {
        'total': AverageMeter(), 'ent': AverageMeter(),
        'height': AverageMeter(), 'pl': AverageMeter(),
    } for s in scale_names}
    total_meter = AverageMeter()

    accs = {s: {'ordering': 0.0, 'cone': 0.0, 'pl': 0.0} for s in scale_names}
    n_batches = 0

    model_to_use = model.module if hasattr(model, 'module') else model
    _curv = model_to_use.curvature.detach()

    logger.info("***** Running Validation *****")
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):
        # Val loader returns only x_temporal (no contrast views)
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(args.device)
        else:
            x = batch.to(args.device)

        temporal_out, _ = model(x)
        sw = temporal_out['scale_weights']

        batch_total_loss = 0.0
        for idx, sname in enumerate(scale_names):
            h_proj = temporal_out[sname]['h_proj']
            scores = temporal_out[sname]['scores']

            ent_dict = entailment_criterions[sname](h_proj, _curv)
            pl_loss = pl_criterion(scores)

            scale_loss = (args.cone_weight * ent_dict["entailment_loss"]
                          + args.height_weight * ent_dict["height_loss"]
                          + args.pl_weight * pl_loss)

            meters[sname]['total'].update(scale_loss.item())
            meters[sname]['ent'].update(ent_dict["entailment_loss"].item())
            meters[sname]['height'].update(ent_dict["height_loss"].item())
            meters[sname]['pl'].update(pl_loss.item())

            batch_total_loss += sw[idx].item() * scale_loss.item()

            accs[sname]['ordering'] += hyperbolic_ordering_accuracy(h_proj, _curv)
            accs[sname]['cone'] += hyperbolic_cone_accuracy(h_proj, _curv, args.min_radius)
            accs[sname]['pl'] += kendall_tau_accuracy(scores)

        total_meter.update(batch_total_loss)
        n_batches += 1

        epoch_iterator.set_description(f"Validating... (loss={total_meter.val:.5f})")

    logger.info(f"Validation Results - Step: {global_step}")
    logger.info(f"  Total avg loss: {total_meter.avg:.5f}")
    logger.info(f"  Scale weights: full={sw[0].item():.4f}, "
                f"half={sw[1].item():.4f}, quarter={sw[2].item():.4f}")

    log_dict = {"step": global_step, "val/loss": total_meter.avg}
    for idx, sname in enumerate(scale_names):
        ordering_acc = accs[sname]['ordering'] / max(n_batches, 1)
        cone_acc = accs[sname]['cone'] / max(n_batches, 1)
        pl_acc = accs[sname]['pl'] / max(n_batches, 1)

        logger.info(f"  [{sname}] loss={meters[sname]['total'].avg:.5f} "
                     f"ent={meters[sname]['ent'].avg:.5f} "
                     f"height={meters[sname]['height'].avg:.5f} "
                     f"pl={meters[sname]['pl'].avg:.5f} "
                     f"ord_acc={ordering_acc:.4f} "
                     f"cone_acc={cone_acc:.4f} "
                     f"pl_acc={pl_acc:.4f}")

        writer.add_scalar(f"val/{sname}/loss", meters[sname]['total'].avg, global_step)
        writer.add_scalar(f"val/{sname}/ordering_accuracy", ordering_acc, global_step)
        writer.add_scalar(f"val/{sname}/cone_accuracy", cone_acc, global_step)
        writer.add_scalar(f"val/{sname}/pl_pair_accuracy", pl_acc, global_step)

        log_dict[f"val/{sname}/loss"] = meters[sname]['total'].avg
        log_dict[f"val/{sname}/ordering_accuracy"] = ordering_acc
        log_dict[f"val/{sname}/cone_accuracy"] = cone_acc
        log_dict[f"val/{sname}/pl_pair_accuracy"] = pl_acc

    writer.add_scalar("val/loss", total_meter.avg, global_step)

    if args.use_wandb:
        wandb.log(log_dict)

    return total_meter.avg


# =============================================
# Train
# =============================================

def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

        if args.use_wandb:
            if not HAS_WANDB:
                logger.warning("wandb not installed. pip install wandb")
                args.use_wandb = False
            else:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.name,
                    config=vars(args),
                )
                wandb.watch(model, log="gradients", log_freq=100)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Data (angular-aware loader)
    train_loader, test_loader = get_angular_loader(args)

    # Per-scale entailment losses
    scale_names = ['full', 'half', 'quarter']
    entailment_criterions = {
        s: HyperbolicEntailmentLoss(
            min_radius=args.min_radius,
            height_margin=args.height_margin,
            height_weight=1.0,
            cone_weight=1.0,
        )
        for s in scale_names
    }
    pl_criterion = PlackettLuceLoss(
        sample=args.pl_sample,
        R=args.pl_R,
        K=args.pl_K,
    )
    angular_criterion = AngularContrastiveLoss(
        temperature=args.angular_temperature,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    scaler = GradScaler(enabled=args.fp16)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    model_to_use = model.module if hasattr(model, 'module') else model

    logger.info("***** Running Multi-Scale MAT + Angular Contrastive Training *****")
    logger.info(f"  Total optimization steps   = {args.num_steps}")
    logger.info(f"  Batch size per GPU         = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation      = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip)      = {args.seq_len}")
    logger.info(f"  Contrast K                 = {args.contrast_k}")
    logger.info(f"  Embedding dim              = {args.embed_dim}")
    logger.info(f"  Angular weight λ           = {args.angular_weight}")
    logger.info(f"  Angular temperature τ      = {args.angular_temperature}")
    logger.info(f"  Angular proj dim           = {args.angular_proj_dim}")
    logger.info(f"  Loss weights: cone={args.cone_weight}, "
                f"height={args.height_weight}, pl={args.pl_weight}")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    angular_losses = AverageMeter()
    angular_accs = AverageMeter()
    global_step, best_loss = 0, float('inf')

    while True:
        model.train()

        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(global_step // max(len(train_loader), 1))
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(global_step // max(len(train_loader), 1))

        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training ({global_step}/{t_total}) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=args.local_rank not in [-1, 0],
        )

        for step, batch in enumerate(epoch_iterator):
            x_temporal, x_contrast, contrast_indices = batch
            x_temporal = x_temporal.to(args.device)
            x_contrast = x_contrast.to(args.device)
            contrast_indices = contrast_indices.to(args.device)

            with autocast(enabled=args.fp16):
                # Forward both branches
                temporal_out, angular_out = model(
                    x_temporal, x_contrast, contrast_indices
                )
                _curv = model_to_use.curvature
                sw = temporal_out['scale_weights']

                # ── Radial branch: multi-scale ordering loss ──
                ordering_loss = torch.tensor(0.0, device=x_temporal.device)
                per_scale_losses = {}

                for idx, sname in enumerate(scale_names):
                    h_proj = temporal_out[sname]['h_proj']
                    scores = temporal_out[sname]['scores']

                    ent_dict = entailment_criterions[sname](h_proj, _curv)
                    pl_loss = pl_criterion(scores)

                    scale_loss = (args.cone_weight * ent_dict["entailment_loss"]
                                  + args.height_weight * ent_dict["height_loss"]
                                  + args.pl_weight * pl_loss)

                    ordering_loss = ordering_loss + sw[idx] * scale_loss

                    per_scale_losses[sname] = {
                        'ent': ent_dict["entailment_loss"].item(),
                        'height': ent_dict["height_loss"].item(),
                        'pl': pl_loss.item(),
                        'total': scale_loss.item(),
                    }

                # ── Angular branch: contrastive loss ──
                angular_loss_val = torch.tensor(0.0, device=x_temporal.device)
                angular_acc_val = 0.0

                if angular_out is not None:
                    angular_loss_val, angular_acc_val = angular_criterion(
                        angular_out['z_temporal'],
                        angular_out['z_contrast'],
                    )

                # ── Total loss ──
                loss = ordering_loss + args.angular_weight * angular_loss_val

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                angular_losses.update(angular_loss_val.item())
                angular_accs.update(angular_acc_val.item() if isinstance(angular_acc_val, float)
                                    else angular_acc_val.item())

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Gradient norm for logging
                if args.local_rank in [-1, 0] and args.use_wandb:
                    total_grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    f"Training ({global_step}/{t_total}) "
                    f"(loss={losses.val:.4f} ang={angular_losses.val:.4f})"
                )

                if args.local_rank in [-1, 0]:
                    sw_vals = sw.detach()

                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/ordering_loss",
                                      ordering_loss.item(), global_step)
                    writer.add_scalar("train/angular_loss",
                                      angular_loss_val.item(), global_step)
                    writer.add_scalar("train/angular_acc",
                                      angular_accs.val, global_step)
                    writer.add_scalar("train/lr",
                                      scheduler.get_lr()[0], global_step)
                    writer.add_scalar("train/curvature",
                                      model_to_use.curvature.item(), global_step)
                    writer.add_scalar("train/scale_weight_full",
                                      sw_vals[0].item(), global_step)
                    writer.add_scalar("train/scale_weight_half",
                                      sw_vals[1].item(), global_step)
                    writer.add_scalar("train/scale_weight_quarter",
                                      sw_vals[2].item(), global_step)

                    for sname in scale_names:
                        writer.add_scalar(f"train/{sname}/entailment_loss",
                                          per_scale_losses[sname]['ent'], global_step)
                        writer.add_scalar(f"train/{sname}/height_loss",
                                          per_scale_losses[sname]['height'], global_step)
                        writer.add_scalar(f"train/{sname}/pl_loss",
                                          per_scale_losses[sname]['pl'], global_step)

                    if args.use_wandb:
                        log_dict = {
                            "train/loss": losses.val,
                            "train/loss_avg": losses.avg,
                            "train/ordering_loss": ordering_loss.item(),
                            "train/angular_loss": angular_loss_val.item(),
                            "train/angular_acc": angular_accs.val,
                            "train/lr": scheduler.get_lr()[0],
                            "train/curvature": model_to_use.curvature.item(),
                            "train/alpha": model_to_use.encoder.lorentz_proj.alpha.exp().item(),
                            "train/grad_norm": total_grad_norm,
                            "train/scale_weight_full": sw_vals[0].item(),
                            "train/scale_weight_half": sw_vals[1].item(),
                            "train/scale_weight_quarter": sw_vals[2].item(),
                            "train/scale_weight_temp": model_to_use.scale_weights.temperature.item(),
                            "step": global_step,
                        }

                        # Angular stats
                        if angular_out is not None:
                            for k, v in angular_out['stats'].items():
                                log_dict[f"train/angular/{k}"] = v

                        # Per-scale losses
                        for sname in scale_names:
                            for k, v in per_scale_losses[sname].items():
                                log_dict[f"train/{sname}/{k}_loss"] = v

                        # Embedding norms
                        with torch.no_grad():
                            h_full_norms = temporal_out['full']['h_proj'].float().norm(dim=-1)
                            log_dict["train/embed_norm_full_mean"] = h_full_norms.mean().item()
                            log_dict["train/embed_norm_full_max"] = h_full_norms.max().item()

                        wandb.log(log_dict)

                # Validation
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    if test_loader is not None:
                        val_loss = valid(
                            args, model, entailment_criterions, pl_criterion,
                            writer, test_loader, global_step,
                        )
                        if val_loss < best_loss:
                            save_model(args, model, step=global_step)
                            best_loss = val_loss
                    else:
                        save_model(args, model, step=global_step)
                    model.train()

                if global_step >= t_total:
                    break

        losses.reset()
        angular_losses.reset()
        angular_accs.reset()
        if global_step >= t_total:
            break

    # Final save
    if args.local_rank in [-1, 0]:
        save_model(args, model, step=global_step)
        writer.close()
        if args.use_wandb:
            wandb.finish()

    logger.info(f"Best validation loss: {best_loss:.5f}")
    logger.info("End Training!")


# =============================================
# Main
# =============================================

def main():
    parser = argparse.ArgumentParser()

    # Names / paths
    parser.add_argument("--name", required=True)
    parser.add_argument("--model_type",
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16")
    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--output_dir", default="output", type=str)

    # Dataset
    parser.add_argument("--dataset", type=str, default="cholec80",
                        choices=["cholec80", "lemon"],
                        help="Dataset to use for training (default: cholec80).")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Training data root. If None, uses default for --dataset.")
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--seq_len", default=8, type=int)
    parser.add_argument("--min_step", default=1, type=int)
    parser.add_argument("--max_step", default=20, type=int)
    parser.add_argument("--sampling_mode", default="randstep",
                        choices=["randstep", "global"])

    # Hyperbolic (MERU-style)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--curv_init", default=1.0, type=float)
    parser.add_argument("--learn_curv", action="store_true", default=True)
    parser.add_argument("--no_learn_curv", dest="learn_curv", action="store_false")

    # Pre-split Lorentz interaction
    parser.add_argument("--pre_split_n_layers", default=2, type=int)
    parser.add_argument("--pre_split_n_heads", default=4, type=int)
    parser.add_argument("--pre_split_mlp_ratio", default=4.0, type=float)
    parser.add_argument("--pre_split_dropout", default=0.1, type=float)

    # Entailment loss
    parser.add_argument("--min_radius", default=0.1, type=float)
    parser.add_argument("--height_margin", default=0.1, type=float)
    parser.add_argument("--height_weight", default=1.0, type=float)
    parser.add_argument("--cone_weight", default=1.0, type=float)

    # Plackett-Luce loss
    parser.add_argument("--pl_weight", default=1.0, type=float)
    parser.add_argument("--pl_sample", action="store_true")
    parser.add_argument("--pl_R", default=4, type=int)
    parser.add_argument("--pl_K", default=8, type=int)

    # Learnable scale weights
    parser.add_argument("--scale_weight_temp", default=1.0, type=float)
    parser.add_argument("--scale_min_weight", default=0.01, type=float)

    # Per-scale Lorentz Score Head
    parser.add_argument("--score_n_layers", default=2, type=int)
    parser.add_argument("--score_n_heads", default=4, type=int)
    parser.add_argument("--score_mlp_ratio", default=4.0, type=float)
    parser.add_argument("--score_dropout", default=0.1, type=float)

    # ── Angular Contrastive (NEW) ──
    parser.add_argument("--contrast_k", default=2, type=int,
                        help="Number of frames per clip for contrastive pairs. "
                             "Higher = better contrastive signal but more compute.")
    parser.add_argument("--angular_weight", default=0.5, type=float,
                        help="Weight λ for angular contrastive loss in total loss.")
    parser.add_argument("--angular_temperature", default=0.1, type=float,
                        help="Temperature τ for InfoNCE contrastive loss.")
    parser.add_argument("--angular_proj_dim", default=128, type=int,
                        help="Projection dimension for angular contrastive head.")

    # Training
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--num_steps", default=50000, type=int)
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--fp16", action="store_true")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="hyperbolic-temporal-vit")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    # -- Device --
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # -- Logging --
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, "
        f"n_gpu: {args.n_gpu}, distributed: {args.local_rank != -1}, "
        f"fp16: {args.fp16}"
    )

    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()