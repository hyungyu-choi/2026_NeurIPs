# coding=utf-8
"""
Train ViT with MERU-style hyperbolic entailment loss on Lorentz hyperboloid.

Each frame's CLS token is projected onto the hyperboloid via exp_map at origin.
Earlier frames entail later frames (closer to apex = more general).

Key differences from the previous geoopt-based version:
  - Only space components are stored (time computed on-the-fly)
  - Learnable curvature parameter (stored as log, following MERU)
  - Learnable alpha scaling before exp_map
  - Entailment loss: clamp(oxy_angle - half_aperture, min=0)
  - No geoopt dependency
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
    hyperbolic_ordering_accuracy,
    hyperbolic_cone_accuracy,
)
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
def setup(args):
    config = CONFIGS[args.model_type]

    pretrained_weights = None
    if args.pretrained_dir is not None:
        pretrained_weights = np.load(args.pretrained_dir)
        logger.info(f"Loaded pretrained weights from {args.pretrained_dir}")
    else:
        logger.info("Training from scratch (no pretrained weights)")

    model = HyperbolicTemporalViT(
        config,
        img_size=args.img_size,
        pretrained_weights=pretrained_weights,
        embed_dim=args.embed_dim,
        curv_init=args.curv_init,
        learn_curv=args.learn_curv,
        zero_head=True,
    )
    model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Config: {config}")
    logger.info(f"Training parameters: {args}")
    logger.info(f"Total trainable parameters: {num_params:.1f}M")
    return args, model


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def valid(args, model, criterion, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    model.eval()

    logger.info("***** Running Validation *****")
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    total_ordering_acc = 0.0
    total_cone_acc = 0.0
    n_batches = 0

    model_to_use = model.module if hasattr(model, 'module') else model
    _curv = model_to_use.curvature.detach()

    for step, batch in enumerate(epoch_iterator):
        x = batch.to(args.device)              # [B, T, 3, H, W]
        h = model(x)                           # [B, T, D] space components
        loss_dict = criterion(h, _curv)
        eval_losses.update(loss_dict["loss"].item())

        total_ordering_acc += hyperbolic_ordering_accuracy(h, _curv)
        total_cone_acc += hyperbolic_cone_accuracy(h, _curv, args.min_radius)
        n_batches += 1

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    ordering_acc = total_ordering_acc / max(n_batches, 1)
    cone_acc = total_cone_acc / max(n_batches, 1)

    logger.info(f"Validation Results - Step: {global_step}")
    logger.info(f"  Loss: {eval_losses.avg:.5f}")
    logger.info(f"  Height ordering accuracy: {ordering_acc:.4f}")
    logger.info(f"  Cone inclusion accuracy:  {cone_acc:.4f}")

    writer.add_scalar("val/loss", eval_losses.avg, global_step)
    writer.add_scalar("val/ordering_accuracy", ordering_acc, global_step)
    writer.add_scalar("val/cone_accuracy", cone_acc, global_step)

    if args.use_wandb:
        wandb.log({
            "val/loss": eval_losses.avg,
            "val/ordering_accuracy": ordering_acc,
            "val/cone_accuracy": cone_acc,
            "step": global_step,
        })

    return eval_losses.avg


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

        # ── Wandb init ──
        if args.use_wandb:
            if not HAS_WANDB:
                logger.warning("wandb not installed. pip install wandb")
                args.use_wandb = False
            else:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.name,
                    config={
                        "model_type": args.model_type,
                        "img_size": args.img_size,
                        "seq_len": args.seq_len,
                        "embed_dim": args.embed_dim,
                        "curv_init": args.curv_init,
                        "learn_curv": args.learn_curv,
                        "min_radius": args.min_radius,
                        "height_margin": args.height_margin,
                        "height_weight": args.height_weight,
                        "cone_weight": args.cone_weight,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "num_steps": args.num_steps,
                        "warmup_steps": args.warmup_steps,
                        "max_grad_norm": args.max_grad_norm,
                        "train_batch_size": args.train_batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "sampling_mode": args.sampling_mode,
                        "min_step": args.min_step,
                        "max_step": args.max_step,
                        "fp16": args.fp16,
                        "seed": args.seed,
                    },
                )
                wandb.watch(model, log="gradients", log_freq=100)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Data
    train_loader, test_loader = get_loader(args)

    # Loss (no learnable params — curvature comes from model)
    criterion = HyperbolicEntailmentLoss(
        min_radius=args.min_radius,
        height_margin=args.height_margin,
        height_weight=args.height_weight,
        cone_weight=args.cone_weight,
    )

    # Optimizer (model params include curvature & alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # AMP
    scaler = GradScaler(enabled=args.fp16)

    # DDP
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # ── Training loop ──
    model_to_use = model.module if hasattr(model, 'module') else model

    logger.info("***** Running MERU-style Hyperbolic Entailment Training *****")
    logger.info(f"  Total optimization steps = {args.num_steps}")
    logger.info(f"  Batch size per GPU       = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip)    = {args.seq_len}")
    logger.info(f"  Embedding dim            = {args.embed_dim}")
    logger.info(f"  Initial curvature        = {args.curv_init}")
    logger.info(f"  Learn curvature          = {args.learn_curv}")

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

        epoch_iterator = tqdm(train_loader,
                              desc=f"Training ({global_step}/{t_total}) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            x = batch.to(args.device)  # [B, T, 3, H, W]

            with autocast(enabled=args.fp16):
                h = model(x)                           # [B, T, D] space components
                _curv = model_to_use.curvature
                loss_dict = criterion(h, _curv)
                loss = loss_dict["loss"]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Compute gradient norm BEFORE zero_grad
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
                                      loss_dict["entailment_loss"].item(), global_step)
                    writer.add_scalar("train/height_loss",
                                      loss_dict["height_loss"].item(), global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("train/curvature",
                                      model_to_use.curvature.item(), global_step)

                    if args.use_wandb:
                        # Embedding stats (space component norms)
                        with torch.no_grad():
                            h_norms = h.float().norm(dim=-1)  # (B, T)

                        wandb.log({
                            "train/loss": losses.val,
                            "train/loss_avg": losses.avg,
                            "train/entailment_loss": loss_dict["entailment_loss"].item(),
                            "train/height_loss": loss_dict["height_loss"].item(),
                            "train/lr": scheduler.get_lr()[0],
                            "train/curvature": model_to_use.curvature.item(),
                            "train/alpha": model_to_use.lorentz_proj.alpha.exp().item(),
                            "train/grad_norm": total_grad_norm,
                            "train/embed_norm_mean": h_norms.mean().item(),
                            "train/embed_norm_max": h_norms.max().item(),
                            "train/embed_norm_min": h_norms.min().item(),
                            "step": global_step,
                        })

                # Validation
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    if test_loader is not None:
                        val_loss = valid(args, model, criterion, writer,
                                         test_loader, global_step)
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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    # Names / paths
    parser.add_argument("--name", required=True, help="Run name for logging.")
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
                        help="Dimension of hyperbolic embeddings (space components).")
    parser.add_argument("--curv_init", default=1.0, type=float,
                        help="Initial curvature (positive). Hyperboloid curvature = -curv.")
    parser.add_argument("--learn_curv", action="store_true", default=True,
                        help="Learn the curvature parameter during training.")
    parser.add_argument("--no_learn_curv", dest="learn_curv", action="store_false",
                        help="Fix the curvature parameter.")

    # Entailment loss
    parser.add_argument("--min_radius", default=0.1, type=float,
                        help="Min radius for half-aperture computation (MERU K parameter).")
    parser.add_argument("--height_margin", default=0.1, type=float,
                        help="Margin for height ordering loss.")
    parser.add_argument("--height_weight", default=1.0, type=float,
                        help="Weight for height ordering loss.")
    parser.add_argument("--cone_weight", default=1.0, type=float,
                        help="Weight for entailment cone loss.")

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
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-temporal-vit",
                        help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (team or username).")

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