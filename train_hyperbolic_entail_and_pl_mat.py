# coding=utf-8
"""
Train ViT with combined MERU-style hyperbolic entailment loss + Plackett-Luce
temporal ordering loss at MULTIPLE SCALES (MAT = Multi-scale Attention on Tangent).

Architecture (v2 - with pre-split Lorentz interaction):

    ViT backbone -> CLS tokens -> MERUStyleProjection (-> hyperboloid, dim D)
                                        |
                              LorentzBlock x N  (Lorentzian attention at full dim D)
                              [frame interaction ON the hyperboloid before splitting]
                                        |
                          HyperbolicDimReduction -> dim D/2 (hyperboloid)
                          HyperbolicDimReduction -> dim D/4 (hyperboloid)
                                        |
                     For each scale (D, D/2, D/4):
                        LorentzScoreHead -> PL scores + refined h
                        Entailment loss + PL loss
                                        |
                     total_loss = sum_s  learned_weight_s * loss_s
                     (scale weights are learnable via softmax over logits)

Key design:
    - Pre-split LorentzBlocks use Lorentzian inner-product attention so that
      frame interaction respects the hyperbolic hierarchy. Q/K are projected
      onto the hyperboloid and scored via <q,k>_L, which encodes both
      direction similarity AND hierarchical position (depth on the hyperboloid).
    - This enriches embeddings with inter-frame context BEFORE dimension
      reduction, so each scale inherits the full-dimensional relational info.
    - Scale loss weights are learned (softmax-normalized logits) so the model
      discovers which granularity matters most for the task.

Hyperbolic-preserving dimension reduction:
    log_map0 -> LayerNorm -> Linear -> GELU -> Linear -> exp_map0
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
from models import lorentz_ops as L
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)


# =============================================
# Hyperbolic Dimension Reduction
# =============================================

class HyperbolicDimReduction(nn.Module):
    """
    Reduce embedding dimension while staying on the Lorentz hyperboloid.

    Flow:
        h_space (B, D_in) on hyperboloid
            -> log_map0  -> tangent vector (B, D_in)   [Euclidean]
            -> LayerNorm -> Linear(D_in, D_out) -> GELU -> Linear(D_out, D_out)
            -> exp_map0  -> h_space (B, D_out)          [hyperboloid]
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        orig_shape = h.shape[:-1]
        D_in = h.shape[-1]
        h_flat = h.reshape(-1, D_in)

        if h_flat.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float32):
                h_tan = L.log_map0(h_flat.float(), curv)
        else:
            h_tan = L.log_map0(h_flat.float(), curv)

        h_tan = self.norm(h_tan)
        h_tan = self.fc1(h_tan)
        h_tan = self.act(h_tan)
        h_tan = self.fc2(h_tan)

        if h_flat.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float32):
                h_out = L.exp_map0(h_tan.float(), curv)
        else:
            h_out = L.exp_map0(h_tan.float(), curv)

        return h_out.reshape(*orig_shape, self.out_dim)


# =============================================
# Learnable Scale Weights
# =============================================

class LearnableScaleWeights(nn.Module):
    """
    Learnable weights for combining multi-scale losses.

    Maintains 3 logits (one per scale) and normalizes them via softmax
    with a learnable temperature parameter. This lets the model learn
    which scale to emphasize during training.

    The temperature controls how peaked the distribution is:
      - High temp -> near-uniform weights
      - Low temp  -> winner-take-all
    """

    def __init__(self, n_scales: int = 3, init_temp: float = 1.0):
        super().__init__()
        # Initialize logits to 0 -> equal weights after softmax
        self.logits = nn.Parameter(torch.zeros(n_scales))
        self.log_temp = nn.Parameter(torch.tensor(init_temp).log())

    def forward(self) -> torch.Tensor:
        """Returns (n_scales,) tensor of positive weights summing to 1."""
        temp = self.log_temp.exp().clamp(min=0.01, max=10.0)
        return F.softmax(self.logits / temp, dim=0)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=10.0)


# =============================================
# Multi-Scale Combined Model (v2)
# =============================================

class MultiScaleHyperbolicCombinedModel(nn.Module):
    """
    ViT -> MERUStyleProjection -> [Pre-split LorentzBlocks] -> Multi-scale split
    -> Per-scale LorentzScoreHead -> scores + refined embeddings

    The pre-split LorentzBlocks let frames interact on the hyperboloid at
    full dimensionality BEFORE splitting, so inter-frame relational info
    is baked into the embeddings that get reduced to D/2 and D/4.
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
        zero_head: bool = True,
        vis: bool = False,
    ):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

        dim_half = embed_dim // 2
        dim_quarter = embed_dim // 4

        # -- Encoder: ViT -> hyperboloid (dim D) --
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

        # -- Pre-split Lorentz interaction blocks (full dim D) --
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

        # -- Hyperbolic dimension reductions --
        self.reduce_half = HyperbolicDimReduction(embed_dim, dim_half)
        self.reduce_quarter = HyperbolicDimReduction(embed_dim, dim_quarter)

        # -- Per-scale Lorentz Score Heads --
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

        # -- Learnable scale weights --
        self.scale_weights = LearnableScaleWeights(
            n_scales=3, init_temp=scale_weight_temp,
        )

        self.embed_dim = embed_dim
        self.dim_half = dim_half
        self.dim_quarter = dim_quarter

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, 3, H, W)
        Returns:
            dict with 'full', 'half', 'quarter' sub-dicts and 'scale_weights'
        """
        curv = self.encoder.curvature

        # -- Full-scale hyperbolic embeddings --
        h = self.encoder(x)  # (B, T, D) on hyperboloid

        # -- Pre-split Lorentz interaction (on hyperboloid at full D) --
        for block in self.pre_split_blocks:
            h = block(h, curv)
        # h is still (B, T, D) on hyperboloid, now enriched with
        # inter-frame context via Lorentzian attention

        h_full = h

        # -- Reduce to half and quarter (on hyperboloid) --
        h_half = self.reduce_half(h_full, curv)        # (B, T, D/2)
        h_quarter = self.reduce_quarter(h_full, curv)  # (B, T, D/4)

        # -- Score heads at each scale --
        scores_full, h_ref_full = self.score_head_full(h_full, curv)
        scores_half, h_ref_half = self.score_head_half(h_half, curv)
        scores_quarter, h_ref_quarter = self.score_head_quarter(h_quarter, curv)

        # -- Learnable scale weights --
        sw = self.scale_weights()  # (3,)

        return {
            'full': {
                'h_proj': h_full,
                'scores': scores_full,
                'h_ref': h_ref_full,
            },
            'half': {
                'h_proj': h_half,
                'scores': scores_half,
                'h_ref': h_ref_half,
            },
            'quarter': {
                'h_proj': h_quarter,
                'scores': scores_quarter,
                'h_ref': h_ref_quarter,
            },
            'scale_weights': sw,
        }

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
    """Fraction of correctly-ordered pairs (higher score = earlier frame)."""
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

    model = MultiScaleHyperbolicCombinedModel(
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
        zero_head=True,
    )
    model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Config: {config}")
    logger.info(f"Training parameters: {args}")
    logger.info(f"Total trainable parameters: {num_params:.1f}M")
    logger.info(f"Multi-scale dims: full={args.embed_dim}, "
                f"half={args.embed_dim // 2}, quarter={args.embed_dim // 4}")
    logger.info(f"Pre-split LorentzBlocks: {args.pre_split_n_layers} layers, "
                f"{args.pre_split_n_heads} heads")
    logger.info(f"Scale weights: LEARNABLE (init_temp={args.scale_weight_temp})")
    return args, model


# =============================================
# Validation
# =============================================

@torch.no_grad()
def valid(args, model, entailment_criterions, pl_criterion, writer,
          test_loader, global_step):
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
        x = batch.to(args.device)
        out = model(x)
        sw = out['scale_weights']  # (3,) learned weights

        batch_total_loss = 0.0
        for idx, sname in enumerate(scale_names):
            h_proj = out[sname]['h_proj']
            scores = out[sname]['scores']

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
        writer.add_scalar(f"val/{sname}/entailment_loss", meters[sname]['ent'].avg, global_step)
        writer.add_scalar(f"val/{sname}/height_loss", meters[sname]['height'].avg, global_step)
        writer.add_scalar(f"val/{sname}/pl_loss", meters[sname]['pl'].avg, global_step)
        writer.add_scalar(f"val/{sname}/ordering_accuracy", ordering_acc, global_step)
        writer.add_scalar(f"val/{sname}/cone_accuracy", cone_acc, global_step)
        writer.add_scalar(f"val/{sname}/pl_pair_accuracy", pl_acc, global_step)

        log_dict.update({
            f"val/{sname}/loss": meters[sname]['total'].avg,
            f"val/{sname}/entailment_loss": meters[sname]['ent'].avg,
            f"val/{sname}/height_loss": meters[sname]['height'].avg,
            f"val/{sname}/pl_loss": meters[sname]['pl'].avg,
            f"val/{sname}/ordering_accuracy": ordering_acc,
            f"val/{sname}/cone_accuracy": cone_acc,
            f"val/{sname}/pl_pair_accuracy": pl_acc,
        })

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

    # Data
    train_loader, test_loader = get_loader(args)

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
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    model_to_use = model.module if hasattr(model, 'module') else model

    logger.info("***** Running Multi-Scale Hyperbolic Entailment + PL Training (v2) *****")
    logger.info(f"  Total optimization steps   = {args.num_steps}")
    logger.info(f"  Batch size per GPU         = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation      = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip)      = {args.seq_len}")
    logger.info(f"  Embedding dim              = {args.embed_dim} "
                f"(scales: {args.embed_dim}, {args.embed_dim//2}, {args.embed_dim//4})")
    logger.info(f"  Pre-split LorentzBlocks    = {args.pre_split_n_layers} layers")
    logger.info(f"  Score head layers          = {args.score_n_layers}")
    logger.info(f"  Loss weights: cone={args.cone_weight}, "
                f"height={args.height_weight}, pl={args.pl_weight}")
    logger.info(f"  Scale weights: LEARNABLE (softmax, temp_init={args.scale_weight_temp})")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
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
            x = batch.to(args.device)  # [B, T, 3, H, W]

            with autocast(enabled=args.fp16):
                out = model(x)
                _curv = model_to_use.curvature
                sw = out['scale_weights']  # (3,) learned weights

                # Accumulate loss over all scales with learned weights
                total_loss = torch.tensor(0.0, device=x.device)
                per_scale_losses = {}

                for idx, sname in enumerate(scale_names):
                    h_proj = out[sname]['h_proj']
                    scores = out[sname]['scores']

                    ent_dict = entailment_criterions[sname](h_proj, _curv)
                    pl_loss = pl_criterion(scores)

                    scale_loss = (args.cone_weight * ent_dict["entailment_loss"]
                                  + args.height_weight * ent_dict["height_loss"]
                                  + args.pl_weight * pl_loss)

                    # Use LEARNED weight for this scale
                    total_loss = total_loss + sw[idx] * scale_loss

                    per_scale_losses[sname] = {
                        'ent': ent_dict["entailment_loss"].item(),
                        'height': ent_dict["height_loss"].item(),
                        'pl': pl_loss.item(),
                        'total': scale_loss.item(),
                    }

                loss = total_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

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
                    f"Training ({global_step}/{t_total}) (loss={losses.val:.5f})"
                )

                if args.local_rank in [-1, 0]:
                    # Current learned scale weights
                    sw_vals = sw.detach()

                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("train/curvature",
                                      model_to_use.curvature.item(), global_step)
                    writer.add_scalar("train/scale_weight_full",
                                      sw_vals[0].item(), global_step)
                    writer.add_scalar("train/scale_weight_half",
                                      sw_vals[1].item(), global_step)
                    writer.add_scalar("train/scale_weight_quarter",
                                      sw_vals[2].item(), global_step)
                    writer.add_scalar("train/scale_weight_temp",
                                      model_to_use.scale_weights.temperature.item(),
                                      global_step)

                    for sname in scale_names:
                        writer.add_scalar(f"train/{sname}/entailment_loss",
                                          per_scale_losses[sname]['ent'], global_step)
                        writer.add_scalar(f"train/{sname}/height_loss",
                                          per_scale_losses[sname]['height'], global_step)
                        writer.add_scalar(f"train/{sname}/pl_loss",
                                          per_scale_losses[sname]['pl'], global_step)
                        writer.add_scalar(f"train/{sname}/total_loss",
                                          per_scale_losses[sname]['total'], global_step)

                    if args.use_wandb:
                        with torch.no_grad():
                            h_full_norms = out['full']['h_proj'].float().norm(dim=-1)
                            h_half_norms = out['half']['h_proj'].float().norm(dim=-1)
                            h_quarter_norms = out['quarter']['h_proj'].float().norm(dim=-1)

                        log_dict = {
                            "train/loss": losses.val,
                            "train/loss_avg": losses.avg,
                            "train/lr": scheduler.get_lr()[0],
                            "train/curvature": model_to_use.curvature.item(),
                            "train/alpha": model_to_use.encoder.lorentz_proj.alpha.exp().item(),
                            "train/grad_norm": total_grad_norm,
                            "train/scale_weight_full": sw_vals[0].item(),
                            "train/scale_weight_half": sw_vals[1].item(),
                            "train/scale_weight_quarter": sw_vals[2].item(),
                            "train/scale_weight_temp": model_to_use.scale_weights.temperature.item(),
                            "train/embed_norm_full_mean": h_full_norms.mean().item(),
                            "train/embed_norm_full_max": h_full_norms.max().item(),
                            "train/embed_norm_half_mean": h_half_norms.mean().item(),
                            "train/embed_norm_half_max": h_half_norms.max().item(),
                            "train/embed_norm_quarter_mean": h_quarter_norms.mean().item(),
                            "train/embed_norm_quarter_max": h_quarter_norms.max().item(),
                            "step": global_step,
                        }
                        for sname in scale_names:
                            log_dict[f"train/{sname}/entailment_loss"] = per_scale_losses[sname]['ent']
                            log_dict[f"train/{sname}/height_loss"] = per_scale_losses[sname]['height']
                            log_dict[f"train/{sname}/pl_loss"] = per_scale_losses[sname]['pl']
                            log_dict[f"train/{sname}/total_loss"] = per_scale_losses[sname]['total']

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
    parser.add_argument("--data_root", type=str,
                        default="../code/Dataset/cholec80/frames/extract_1fps/training_set")
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--seq_len", default=8, type=int)
    parser.add_argument("--min_step", default=1, type=int)
    parser.add_argument("--max_step", default=20, type=int)
    parser.add_argument("--sampling_mode", default="randstep",
                        choices=["randstep", "global"])

    # Hyperbolic (MERU-style)
    parser.add_argument("--embed_dim", default=128, type=int,
                        help="Full-scale hyperbolic embedding dim (must be divisible by 4).")
    parser.add_argument("--curv_init", default=1.0, type=float)
    parser.add_argument("--learn_curv", action="store_true", default=True)
    parser.add_argument("--no_learn_curv", dest="learn_curv", action="store_false")

    # Pre-split Lorentz interaction
    parser.add_argument("--pre_split_n_layers", default=2, type=int,
                        help="Number of LorentzBlock layers before multi-scale split.")
    parser.add_argument("--pre_split_n_heads", default=4, type=int,
                        help="Number of attention heads in pre-split LorentzBlocks.")
    parser.add_argument("--pre_split_mlp_ratio", default=4.0, type=float,
                        help="MLP hidden dim ratio in pre-split LorentzBlocks.")
    parser.add_argument("--pre_split_dropout", default=0.1, type=float,
                        help="Dropout in pre-split LorentzBlocks.")

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
    parser.add_argument("--scale_weight_temp", default=1.0, type=float,
                        help="Initial temperature for scale weight softmax. "
                             "Higher = more uniform initial weights.")

    # Per-scale Lorentz Score Head
    parser.add_argument("--score_n_layers", default=2, type=int)
    parser.add_argument("--score_n_heads", default=4, type=int)
    parser.add_argument("--score_mlp_ratio", default=4.0, type=float)
    parser.add_argument("--score_dropout", default=0.1, type=float)

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