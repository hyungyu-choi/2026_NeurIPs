# coding=utf-8
"""
Train ViT for self-supervised temporal frame ordering (Plackett-Luce loss).
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
from models.temporal_vit import TemporalViT, PlackettLuceLoss
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

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


def kendall_tau_accuracy(logits, seq_len):
    """
    Fraction of correctly-ordered pairs.
    logits: [B, T]  (higher → earlier)
    Ground truth order: 0, 1, ..., T-1  (descending logit = correct)
    """
    B, T = logits.shape
    pred_order = torch.argsort(logits, dim=1, descending=True)  # predicted earliest→latest

    correct_pairs = 0
    total_pairs = 0
    for b in range(B):
        po = pred_order[b]
        for i in range(T):
            for j in range(i + 1, T):
                # ground truth: frame po[i] should come before po[j]
                if po[i] < po[j]:
                    correct_pairs += 1
                total_pairs += 1
    return correct_pairs / max(total_pairs, 1)


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

    model = TemporalViT(
        config,
        img_size=args.img_size,
        pretrained_weights=pretrained_weights,
        hidden_mul=args.hidden_mul,
        max_len=args.temporal_max_len,
        n_layers=args.temporal_n_layers,
        n_head=args.temporal_n_head,
        dropout=args.temporal_dropout,
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

    total_correct_pairs = 0
    total_pairs = 0

    for step, batch in enumerate(epoch_iterator):
        x = batch.to(args.device)                    # [B, T, 3, H, W]
        logits = model(x)                             # [B, T]
        loss = criterion(logits)
        eval_losses.update(loss.item())

        # Pairwise ordering accuracy
        B, T = logits.shape
        pred_order = torch.argsort(logits, dim=1, descending=True)
        for b in range(B):
            po = pred_order[b]
            for i in range(T):
                for j in range(i + 1, T):
                    if po[i] < po[j]:
                        total_correct_pairs += 1
                    total_pairs += 1

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    accuracy = total_correct_pairs / max(total_pairs, 1)

    logger.info(f"Validation Results - Step: {global_step}")
    logger.info(f"  Loss: {eval_losses.avg:.5f}")
    logger.info(f"  Pairwise ordering accuracy: {accuracy:.4f}")

    writer.add_scalar("val/loss", eval_losses.avg, global_step)
    writer.add_scalar("val/pair_accuracy", accuracy, global_step)
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

    # Loss
    criterion = PlackettLuceLoss(
        sample=args.pl_sample,
        R=args.pl_R,
        K=args.pl_K,
    )

    # Optimizer & Scheduler
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

    # AMP scaler
    scaler = GradScaler(enabled=args.fp16)

    # DDP
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # ── Training loop ──
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.num_steps}")
    logger.info(f"  Batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Seq len (frames/clip) = {args.seq_len}")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_loss = 0, float('inf')

    while True:
        model.train()

        # Resample temporal clips each outer epoch
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
                logits = model(x)       # [B, T]
                loss = criterion(logits)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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

                # Validation
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    if test_loader is not None:
                        val_loss = valid(args, model, criterion, writer, test_loader, global_step)
                        if val_loss < best_loss:
                            save_model(args, model, step=global_step)
                            best_loss = val_loss
                    else:
                        # No val set → save periodically
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
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Path to pretrained ViT .npz weights.")
    parser.add_argument("--output_dir", default="output", type=str)

    # Dataset
    parser.add_argument("--data_root", type=str,
                        default="../code/Dataset/cholec80/frames/extract_1fps/training_set",
                        help="Path to training video folders.")
    parser.add_argument("--val_root", type=str, default=None,
                        help="Path to validation video folders (optional).")
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--seq_len", default=8, type=int,
                        help="Number of frames per temporal clip.")
    parser.add_argument("--min_step", default=1, type=int,
                        help="Min step for randstep sampling.")
    parser.add_argument("--max_step", default=20, type=int,
                        help="Max step for randstep sampling.")
    parser.add_argument("--sampling_mode", default="randstep",
                        choices=["randstep", "global"])

    # Temporal Head
    parser.add_argument("--hidden_mul", default=0.5, type=float,
                        help="Hidden dim multiplier for TemporalHead (hidden = backbone_dim * hidden_mul).")
    parser.add_argument("--temporal_max_len", default=64, type=int,
                        help="Max sequence length for TemporalSideContext.")
    parser.add_argument("--temporal_n_layers", default=6, type=int,
                        help="Number of TransformerEncoder layers in TemporalHead.")
    parser.add_argument("--temporal_n_head", default=8, type=int,
                        help="Number of attention heads in TemporalHead.")
    parser.add_argument("--temporal_dropout", default=0.1, type=float,
                        help="Dropout rate in TemporalHead.")

    # Plackett-Luce loss
    parser.add_argument("--pl_sample", action="store_true",
                        help="Use subset sampling in PL loss.")
    parser.add_argument("--pl_R", default=4, type=int,
                        help="Number of random subsets per sample.")
    parser.add_argument("--pl_K", default=8, type=int,
                        help="Subset size for PL loss sampling.")

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
    parser.add_argument("--fp16", action="store_true",
                        help="Use native PyTorch AMP (float16).")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Device setup ──
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