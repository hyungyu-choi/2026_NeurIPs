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


class Cholec80TemporalDataset(Dataset):
    """
    Temporal dataset for cholec80 frames stored in per-video folders.

    Directory structure:
        root/
            01/  (or video01, etc.)
                000000.jpg
                000001.jpg
                ...
            02/
                ...

    Sampling modes:
      - 'randstep': fixed-step local sampling (random start + random step)
      - 'global': divide video into seq_len equal segments, pick 1 frame per segment
    """

    # (min_frames, max_frames, num_repeats_per_video)
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
        transform: Optional[Callable] = None,
        min_step: int = 1,
        max_step: int = 20,
        sampling_mode: str = 'randstep',
    ):
        super().__init__()
        assert seq_len > 0, "seq_len must be positive"
        assert 1 <= min_step <= max_step, "Require 1 <= min_step <= max_step"
        assert sampling_mode in ('randstep', 'global'), \
            f"sampling_mode must be 'randstep' or 'global', got '{sampling_mode}'"

        self.root = root
        self.img_size = img_size
        self.seq_len = seq_len
        self.min_step = min_step
        self.max_step = max_step
        self.sampling_mode = sampling_mode

        # --- Scan video folders and collect frame paths ---
        self.vid2frames: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
        self._scan_videos()

        # --- Transform pipeline ---
        self.frame_tx = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # --- Build initial samples ---
        self._build_samples(seed=0)
        logger.info(
            f"[Cholec80TemporalDataset] mode='{self.sampling_mode}', "
            f"seq_len={self.seq_len}, videos={len(self.vid2frames)}, "
            f"total_samples={len(self.samples)}"
        )

    def _scan_videos(self):
        """Scan root directory for video folders and their frame images."""
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

            # Sort by frame index
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
        """Call once per epoch to resample temporal clips."""
        self._build_samples(seed=epoch)

    def _per_video_repeats(self, n_frames: int) -> int:
        return next(rep for lo, hi, rep in self._RULES if lo <= n_frames < hi)

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        samples: List[List[str]] = []  # each sample is a list of frame paths

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

    def __getitem__(self, idx: int) -> torch.Tensor:
        paths = self.samples[idx]
        frames = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            img = self.frame_tx(img)
            frames.append(img)
        return torch.stack(frames, 0)  # [seq_len, 3, H, W]


class Cholec80ValDataset(Dataset):
    """
    Validation dataset: uniformly samples seq_len frames from each video (deterministic).
    One sample per video.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        seq_len: int = 8,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.seq_len = seq_len

        self.frame_tx = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.samples: List[List[str]] = []
        frame_pattern = re.compile(r'(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)

        video_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        for vid_name in video_dirs:
            vid_path = os.path.join(root, vid_name)
            frames = []
            for fname in os.listdir(vid_path):
                m = frame_pattern.match(fname)
                if m:
                    frame_idx = int(m.group(1))
                    frame_path = os.path.join(vid_path, fname)
                    frames.append((frame_idx, frame_path))
            frames.sort(key=lambda x: x[0])

            if len(frames) < seq_len:
                continue

            # Uniform sampling
            n = len(frames)
            indices = [int(i * n / seq_len) for i in range(seq_len)]
            paths = [frames[i][1] for i in indices]
            self.samples.append(paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        paths = self.samples[idx]
        frames = [self.frame_tx(Image.open(p).convert('RGB')) for p in paths]
        return torch.stack(frames, 0)


def get_loader(args):
    """
    Build train/test data loaders for cholec80 temporal dataset.

    Expected args attributes:
        args.data_root       : path to training video folders
        args.val_root        : path to validation video folders (optional, can be None)
        args.img_size        : image resolution (default 224)
        args.seq_len         : number of frames per clip
        args.min_step        : minimum step for randstep sampling
        args.max_step        : maximum step for randstep sampling
        args.sampling_mode   : 'randstep' or 'global'
        args.train_batch_size
        args.eval_batch_size
        args.local_rank
    """
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # --- Defaults for optional attributes ---
    seq_len = getattr(args, 'seq_len', 8)
    min_step = getattr(args, 'min_step', 1)
    max_step = getattr(args, 'max_step', 20)
    sampling_mode = getattr(args, 'sampling_mode', 'randstep')
    data_root = getattr(args, 'data_root',
                        '../code/Dataset/cholec80/frames/extract_1fps/training_set')
    val_root = getattr(args, 'val_root', None)

    trainset = Cholec80TemporalDataset(
        root=data_root,
        img_size=args.img_size,
        seq_len=seq_len,
        min_step=min_step,
        max_step=max_step,
        sampling_mode=sampling_mode,
    )

    testset = None
    if val_root is not None and args.local_rank in [-1, 0]:
        testset = Cholec80ValDataset(
            root=val_root,
            img_size=args.img_size,
            seq_len=seq_len,
        )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
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