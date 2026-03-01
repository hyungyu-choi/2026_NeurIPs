# coding=utf-8
"""
Train ViT with hyperbolic entailment loss on Lorentz hyperboloid.

Each frame's CLS token is projected onto the hyperboloid.
Earlier frames entail later frames (closer to apex = more general).
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
    path = os.path.join(args.output_dir, f"{args.name}{suffix}_checkpoint.bin")
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
        hyp_dim=args.hyp_dim,
        curvature=args.curvature,
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
    K_val = criterion.K_aperture.item()

    for step, batch in enumerate(epoch_iterator):
        x = batch.to(args.device)              # [B, T, 3, H, W]
        h = model(x)                           # [B, T, D+1]
        loss = criterion(h)
        eval_losses.update(loss.item())

        total_ordering_acc += hyperbolic_ordering_accuracy(h, criterion.manifold)
        total_cone_acc += hyperbolic_cone_accuracy(h, K_val, criterion.manifold)
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
    return eval_losses.avg


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Data
    train_loader, test_loader = get_loader(args)

    # Loss (has learnable K_aperture parameter)
    criterion = HyperbolicEntailmentLoss(
        curvature=args.curvature,
        height_margin=args.height_margin,
        cone_margin=args.cone_margin,
        height_weight=args.height_weight,
        cone_weight=args.cone_weight,
    ).to(args.device)

    # Optimizer: model params + loss params (K_aperture)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
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
    logger.info("***** Running Hyperbolic Entailment Training *****")
    logger.info(f"  Total optimization steps = {args.num_steps}")
    logger.info(f"  Batch size per GPU       = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip)    = {args.seq_len}")
    logger.info(f"  Hyperbolic dim           = {args.hyp_dim}")
    logger.info(f"  Curvature                = {args.curvature}")

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
                h = model(x)            # [B, T, D+1]
                loss = criterion(h)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    args.max_grad_norm,
                )

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
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("train/K_aperture",
                                      criterion.K_aperture.item(), global_step)

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
        # Save loss params (K_aperture) separately
        loss_path = os.path.join(args.output_dir, f"{args.name}_loss_params.bin")
        torch.save(criterion.state_dict(), loss_path)
        writer.close()

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

    # Hyperbolic
    parser.add_argument("--hyp_dim", default=128, type=int,
                        help="Dimension of hyperbolic embeddings (output is hyp_dim+1).")
    parser.add_argument("--curvature", default=1.0, type=float,
                        help="Curvature K for Lorentz model (manifold curvature = -1/K).")

    # Entailment loss
    parser.add_argument("--height_margin", default=0.1, type=float,
                        help="Margin for height ordering loss.")
    parser.add_argument("--cone_margin", default=0.05, type=float,
                        help="Margin for cone inclusion loss.")
    parser.add_argument("--height_weight", default=1.0, type=float,
                        help="Weight for height ordering loss.")
    parser.add_argument("--cone_weight", default=1.0, type=float,
                        help="Weight for cone inclusion loss.")

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