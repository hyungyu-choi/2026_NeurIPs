# coding=utf-8
"""
Data utilities for Angular Contrastive + Temporal Ordering training.

Key difference from data_utils.py:
  - Each sample returns TWO augmentations of K randomly chosen frames
    (in addition to the standard temporal clip).
  - The temporal clip uses augmentation pipeline A.
  - The contrastive view uses augmentation pipeline B (different random params).
  - Both pipelines share the same source frames.

Returns per sample:
    x_temporal:       (T, 3, H, W)   — temporal clip with augmentation A
    x_contrast:       (K, 3, H, W)   — contrastive views with augmentation B
    contrast_indices: (K,)            — which positions in [0, T) were selected
"""
import logging
import os
import re
import random
import collections
from typing import List, Tuple, Optional, Callable, Dict

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Re-use dataset configs from the original module
from utils.data_utils import DATASET_CONFIGS, get_dataset_config, TemporalValDataset


class AngularTemporalVideoDataset(Dataset):
    """
    Temporal video dataset that also produces contrastive pairs for
    Angular Contrastive SSL.

    Each __getitem__ returns:
        x_temporal       : (T, 3, H, W)  — augmentation A applied to all T frames
        x_contrast       : (K, 3, H, W)  — augmentation B applied to K selected frames
        contrast_indices : (K,)           — positions in [0, T) that were selected

    The K frames are randomly chosen from the T-frame clip.  Both augmentation
    pipelines start from the SAME raw PIL image for each selected frame.
    """

    _RULES: List[Tuple[int, int, int]] = [
        (0,    240,  60),
        (240,  960,  200),
        (960,  10**9, 300),
    ]

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        seq_len: int = 8,
        contrast_k: int = 2,
        min_step: int = 1,
        max_step: int = 20,
        sampling_mode: str = 'randstep',
        transform_temporal: Optional[Callable] = None,
        transform_contrast: Optional[Callable] = None,
    ):
        super().__init__()
        assert seq_len > 0
        assert 1 <= min_step <= max_step
        assert sampling_mode in ('randstep', 'global')
        assert 1 <= contrast_k <= seq_len, \
            f"contrast_k ({contrast_k}) must be in [1, seq_len ({seq_len})]"

        self.root = root
        self.img_size = img_size
        self.seq_len = seq_len
        self.contrast_k = contrast_k
        self.min_step = min_step
        self.max_step = max_step
        self.sampling_mode = sampling_mode

        # --- Scan video folders ---
        self.vid2frames: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
        self._scan_videos()

        # --- Augmentation pipeline A (temporal ordering) ---
        self.transform_temporal = transform_temporal or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # --- Augmentation pipeline B (contrastive — stronger augmentation) ---
        self.transform_contrast = transform_contrast or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                        saturation=0.3, hue=0.15)],
                p=0.9,
            ),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # --- Build initial samples ---
        self._build_samples(seed=0)
        logger.info(
            f"[AngularTemporalVideoDataset] mode='{self.sampling_mode}', "
            f"seq_len={self.seq_len}, contrast_k={self.contrast_k}, "
            f"videos={len(self.vid2frames)}, total_samples={len(self.samples)}"
        )

    def _scan_videos(self):
        video_dirs = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        frame_pattern = re.compile(r'(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)

        for vid_name in video_dirs:
            vid_path = os.path.join(self.root, vid_name)
            frames = []
            for fname in os.listdir(vid_path):
                m = frame_pattern.match(fname)
                if m:
                    frame_idx = int(m.group(1))
                    frame_path = os.path.join(vid_path, fname)
                    frames.append((frame_idx, frame_path))
            frames.sort(key=lambda x: x[0])

            if len(frames) < self.seq_len:
                logger.warning(
                    f"Video '{vid_name}' has only {len(frames)} frames "
                    f"(< seq_len={self.seq_len}), skipping."
                )
                continue
            self.vid2frames[vid_name] = frames

        logger.info(f"Found {len(self.vid2frames)} videos under '{self.root}'")

    def set_epoch(self, epoch: int):
        self._build_samples(seed=epoch)

    def _per_video_repeats(self, n_frames: int) -> int:
        return next(rep for lo, hi, rep in self._RULES if lo <= n_frames < hi)

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        samples: List[List[str]] = []

        for vid, frame_list in self.vid2frames.items():
            n = len(frame_list)
            if n < self.seq_len:
                continue

            repeats = self._per_video_repeats(n)

            if self.sampling_mode == 'global':
                segment_size = n / self.seq_len
                for _ in range(repeats):
                    positions = []
                    for seg_idx in range(self.seq_len):
                        seg_start = int(seg_idx * segment_size)
                        seg_end = int((seg_idx + 1) * segment_size)
                        seg_end = min(seg_end, n)
                        if seg_start >= seg_end:
                            seg_start = max(0, seg_end - 1)
                        pos = rng.randint(seg_start, seg_end - 1)
                        positions.append(pos)
                    paths = [frame_list[p][1] for p in positions]
                    samples.append(paths)
            else:  # randstep
                min_total_span = self.min_step * (self.seq_len - 1)
                if n - 1 < min_total_span:
                    continue
                for _ in range(repeats):
                    max_start_for_min = n - 1 - min_total_span
                    start_pos = rng.randint(0, max_start_for_min)
                    s_max_feasible = (n - 1 - start_pos) // (self.seq_len - 1)
                    s_max = min(self.max_step, s_max_feasible)
                    step = rng.randint(self.min_step, s_max)
                    positions = [start_pos + i * step for i in range(self.seq_len)]
                    paths = [frame_list[p][1] for p in positions]
                    samples.append(paths)

        self.samples: List[List[str]] = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            x_temporal:       (T, 3, H, W)
            x_contrast:       (K, 3, H, W)
            contrast_indices: (K,)  LongTensor
        """
        paths = self.samples[idx]
        T = len(paths)
        K = self.contrast_k

        # Select K random positions for contrastive pairs
        contrast_pos = sorted(random.sample(range(T), K))

        # Load raw images once, then apply different augmentations
        temporal_frames = []
        contrast_frames = []

        for t, p in enumerate(paths):
            img = Image.open(p).convert('RGB')

            # Augmentation A for temporal clip (all T frames)
            temporal_frames.append(self.transform_temporal(img))

            # Augmentation B for contrastive view (only selected K frames)
            if t in contrast_pos:
                contrast_frames.append(self.transform_contrast(img))

        x_temporal = torch.stack(temporal_frames, 0)          # (T, 3, H, W)
        x_contrast = torch.stack(contrast_frames, 0)          # (K, 3, H, W)
        contrast_indices = torch.tensor(contrast_pos, dtype=torch.long)  # (K,)

        return x_temporal, x_contrast, contrast_indices


def get_angular_loader(args):
    """
    Build train/test data loaders for angular contrastive + temporal ordering.

    Expected args attributes (in addition to standard ones):
        args.contrast_k  : number of frames for contrastive pairs (default: 2)
    """
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Resolve dataset paths
    dataset_name = getattr(args, 'dataset', 'cholec80')
    ds_config = get_dataset_config(dataset_name)

    data_root = getattr(args, 'data_root', None) or ds_config['train_root']
    val_root = getattr(args, 'val_root', None) or ds_config.get('val_root', None)

    logger.info(f"Dataset: {dataset_name} ({ds_config['description']})")
    logger.info(f"  train_root: {data_root}")
    logger.info(f"  val_root:   {val_root}")

    seq_len = getattr(args, 'seq_len', 8)
    min_step = getattr(args, 'min_step', 1)
    max_step = getattr(args, 'max_step', 20)
    sampling_mode = getattr(args, 'sampling_mode', 'randstep')
    contrast_k = getattr(args, 'contrast_k', 2)

    trainset = AngularTemporalVideoDataset(
        root=data_root,
        img_size=args.img_size,
        seq_len=seq_len,
        contrast_k=contrast_k,
        min_step=min_step,
        max_step=max_step,
        sampling_mode=sampling_mode,
    )

    testset = None
    if val_root is not None and args.local_rank in [-1, 0]:
        testset = TemporalValDataset(
            root=val_root,
            img_size=args.img_size,
            seq_len=seq_len,
        )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 \
        else DistributedSampler(trainset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = None
    if testset is not None:
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size,
            num_workers=4,
            pin_memory=True,
        )

    return train_loader, test_loader