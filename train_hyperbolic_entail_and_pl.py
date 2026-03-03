# coding=utf-8
"""
Train ViT with combined MERU-style hyperbolic entailment loss + Plackett-Luce
temporal ordering loss.

Architecture:
    ViT backbone → CLS tokens → MERUStyleProjection (→ hyperboloid)
                                        ↓
                               LorentzScoreHead (Lorentz attention + MLP)
                                    ↓              ↓
                              PL scores (B,T)   refined h (B,T,D)
                                    ↓              ↓
                              PL loss         Entailment loss
                                    ↓              ↓
                              total_loss = cone_w * entailment + height_w * height + pl_w * PL

The LorentzScoreHead processes hyperbolic embeddings through attention layers
whose logits are Lorentzian inner products, preserving the geometric structure.
See models/lorentz_head.py for detailed design notes.
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta

import torch
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
from models.lorentz_head import LorentzScoreHead
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════
# Combined Model
# ═════════════════════════════════════════════

class HyperbolicCombinedModel(torch.nn.Module):
    """
    Wraps HyperbolicTemporalViT + LorentzScoreHead into a single module
    so that DDP handles them together.
    """

    def __init__(
        self,
        config,
        img_size: int = 224,
        pretrained_weights=None,
        embed_dim: int = 128,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        # Score head
        score_n_layers: int = 2,
        score_n_heads: int = 4,
        score_mlp_ratio: float = 4.0,
        score_dropout: float = 0.1,
        zero_head: bool = True,
        vis: bool = False,
    ):
        super().__init__()
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
        self.score_head = LorentzScoreHead(
            embed_dim=embed_dim,
            n_layers=score_n_layers,
            n_heads=score_n_heads,
            mlp_ratio=score_mlp_ratio,
            dropout=score_dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, 3, H, W)
        Returns:
            h_proj:  (B, T, D) initial hyperbolic embeddings (for entailment)
            scores:  (B, T)    PL ordering scores
            h_ref:   (B, T, D) refined hyperbolic embeddings (after score head)
        """
        h_proj = self.encoder(x)                        # (B, T, D) on hyperboloid
        curv = self.encoder.curvature
        scores, h_refined = self.score_head(h_proj, curv)  # scores + refined emb
        return h_proj, scores, h_refined

    @property
    def curvature(self) -> torch.Tensor:
        return self.encoder.curvature


# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════

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
    """
    Fraction of correctly-ordered pairs.
    scores: (B, T) — higher score → earlier frame (ground truth: index 0 is earliest).
    """
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


# ═════════════════════════════════════════════
# Setup
# ═════════════════════════════════════════════

def setup(args):
    config = CONFIGS[args.model_type]

    pretrained_weights = None
    if args.pretrained_dir is not None:
        pretrained_weights = np.load(args.pretrained_dir)
        logger.info(f"Loaded pretrained weights from {args.pretrained_dir}")
    else:
        logger.info("Training from scratch (no pretrained weights)")

    model = HyperbolicCombinedModel(
        config,
        img_size=args.img_size,
        pretrained_weights=pretrained_weights,
        embed_dim=args.embed_dim,
        curv_init=args.curv_init,
        learn_curv=args.learn_curv,
        score_n_layers=args.score_n_layers,
        score_n_heads=args.score_n_heads,
        score_mlp_ratio=args.score_mlp_ratio,
        score_dropout=args.score_dropout,
        zero_head=True,
    )
    model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Config: {config}")
    logger.info(f"Training parameters: {args}")
    logger.info(f"Total trainable parameters: {num_params:.1f}M")
    return args, model


# ═════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════

@torch.no_grad()
def valid(args, model, entailment_criterion, pl_criterion, writer,
          test_loader, global_step):
    model.eval()
    losses_meter = AverageMeter()
    ent_meter = AverageMeter()
    height_meter = AverageMeter()
    pl_meter = AverageMeter()

    total_ordering_acc = 0.0
    total_cone_acc = 0.0
    total_pl_acc = 0.0
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
        h_proj, scores, h_refined = model(x)

        # Entailment loss (on initial projection — cleaner signal)
        ent_dict = entailment_criterion(h_proj, _curv)
        pl_loss = pl_criterion(scores)

        total_loss = (args.cone_weight * ent_dict["entailment_loss"]
                      + args.height_weight * ent_dict["height_loss"]
                      + args.pl_weight * pl_loss)

        losses_meter.update(total_loss.item())
        ent_meter.update(ent_dict["entailment_loss"].item())
        height_meter.update(ent_dict["height_loss"].item())
        pl_meter.update(pl_loss.item())

        # Metrics
        total_ordering_acc += hyperbolic_ordering_accuracy(h_proj, _curv)
        total_cone_acc += hyperbolic_cone_accuracy(h_proj, _curv, args.min_radius)
        total_pl_acc += kendall_tau_accuracy(scores)
        n_batches += 1

        epoch_iterator.set_description(f"Validating... (loss={losses_meter.val:.5f})")

    ordering_acc = total_ordering_acc / max(n_batches, 1)
    cone_acc = total_cone_acc / max(n_batches, 1)
    pl_acc = total_pl_acc / max(n_batches, 1)

    logger.info(f"Validation Results - Step: {global_step}")
    logger.info(f"  Total loss:        {losses_meter.avg:.5f}")
    logger.info(f"  Entailment loss:   {ent_meter.avg:.5f}")
    logger.info(f"  Height loss:       {height_meter.avg:.5f}")
    logger.info(f"  PL loss:           {pl_meter.avg:.5f}")
    logger.info(f"  Height ordering:   {ordering_acc:.4f}")
    logger.info(f"  Cone inclusion:    {cone_acc:.4f}")
    logger.info(f"  PL pair accuracy:  {pl_acc:.4f}")

    writer.add_scalar("val/loss", losses_meter.avg, global_step)
    writer.add_scalar("val/entailment_loss", ent_meter.avg, global_step)
    writer.add_scalar("val/height_loss", height_meter.avg, global_step)
    writer.add_scalar("val/pl_loss", pl_meter.avg, global_step)
    writer.add_scalar("val/ordering_accuracy", ordering_acc, global_step)
    writer.add_scalar("val/cone_accuracy", cone_acc, global_step)
    writer.add_scalar("val/pl_pair_accuracy", pl_acc, global_step)

    if args.use_wandb:
        wandb.log({
            "val/loss": losses_meter.avg,
            "val/entailment_loss": ent_meter.avg,
            "val/height_loss": height_meter.avg,
            "val/pl_loss": pl_meter.avg,
            "val/ordering_accuracy": ordering_acc,
            "val/cone_accuracy": cone_acc,
            "val/pl_pair_accuracy": pl_acc,
            "step": global_step,
        })

    return losses_meter.avg


# ═════════════════════════════════════════════
# Train
# ═════════════════════════════════════════════

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

    # Losses
    entailment_criterion = HyperbolicEntailmentLoss(
        min_radius=args.min_radius,
        height_margin=args.height_margin,
        height_weight=1.0,   # weighting is handled at the total-loss level
        cone_weight=1.0,
    )
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

    logger.info("***** Running Combined Hyperbolic Entailment + PL Training *****")
    logger.info(f"  Total optimization steps   = {args.num_steps}")
    logger.info(f"  Batch size per GPU         = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation      = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip)      = {args.seq_len}")
    logger.info(f"  Embedding dim              = {args.embed_dim}")
    logger.info(f"  Score head layers          = {args.score_n_layers}")
    logger.info(f"  Score head heads           = {args.score_n_heads}")
    logger.info(f"  Loss weights: cone={args.cone_weight}, "
                f"height={args.height_weight}, pl={args.pl_weight}")

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
                h_proj, scores, h_refined = model(x)
                _curv = model_to_use.curvature

                # ── Entailment loss on initial projection ──
                # Using h_proj (not h_refined) gives a cleaner geometric
                # supervision signal to the encoder + projection layer.
                ent_dict = entailment_criterion(h_proj, _curv)

                # ── PL loss on scores from the Lorentz score head ──
                pl_loss = pl_criterion(scores)

                loss = (args.cone_weight * ent_dict["entailment_loss"]
                        + args.height_weight * ent_dict["height_loss"]
                        + args.pl_weight * pl_loss)

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
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/entailment_loss",
                                      ent_dict["entailment_loss"].item(), global_step)
                    writer.add_scalar("train/height_loss",
                                      ent_dict["height_loss"].item(), global_step)
                    writer.add_scalar("train/pl_loss", pl_loss.item(), global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("train/curvature",
                                      model_to_use.curvature.item(), global_step)

                    if args.use_wandb:
                        with torch.no_grad():
                            h_norms = h_proj.float().norm(dim=-1)

                        wandb.log({
                            "train/loss": losses.val,
                            "train/loss_avg": losses.avg,
                            "train/entailment_loss": ent_dict["entailment_loss"].item(),
                            "train/height_loss": ent_dict["height_loss"].item(),
                            "train/pl_loss": pl_loss.item(),
                            "train/lr": scheduler.get_lr()[0],
                            "train/curvature": model_to_use.curvature.item(),
                            "train/alpha": model_to_use.encoder.lorentz_proj.alpha.exp().item(),
                            "train/grad_norm": total_grad_norm,
                            "train/embed_norm_mean": h_norms.mean().item(),
                            "train/embed_norm_max": h_norms.max().item(),
                            "train/embed_norm_min": h_norms.min().item(),
                            "step": global_step,
                        })

                # Validation
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    if test_loader is not None:
                        val_loss = valid(
                            args, model, entailment_criterion, pl_criterion,
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


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════

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
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--curv_init", default=1.0, type=float)
    parser.add_argument("--learn_curv", action="store_true", default=True)
    parser.add_argument("--no_learn_curv", dest="learn_curv", action="store_false")

    # Entailment loss
    parser.add_argument("--min_radius", default=0.1, type=float)
    parser.add_argument("--height_margin", default=0.1, type=float)
    parser.add_argument("--height_weight", default=1.0, type=float)
    parser.add_argument("--cone_weight", default=1.0, type=float)

    # Plackett-Luce loss
    parser.add_argument("--pl_weight", default=1.0, type=float,
                        help="Weight for PL loss in total loss.")
    parser.add_argument("--pl_sample", action="store_true",
                        help="Use subset sampling in PL loss.")
    parser.add_argument("--pl_R", default=4, type=int)
    parser.add_argument("--pl_K", default=8, type=int)

    # Lorentz Score Head
    parser.add_argument("--score_n_layers", default=2, type=int,
                        help="Number of LorentzBlock layers in score head.")
    parser.add_argument("--score_n_heads", default=4, type=int,
                        help="Number of attention heads in LorentzAttention.")
    parser.add_argument("--score_mlp_ratio", default=4.0, type=float,
                        help="MLP hidden dim ratio in LorentzBlock.")
    parser.add_argument("--score_dropout", default=0.1, type=float,
                        help="Dropout in score head.")

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

    # ── Device ──
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # ── Logging ──
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