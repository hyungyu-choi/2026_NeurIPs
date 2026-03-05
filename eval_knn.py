# coding=utf-8
"""
KNN-based Phase Recognition Evaluation.

Extracts CLS token features from a trained ViT backbone (any variant:
TemporalViT, HyperbolicTemporalViT, HyperbolicCombinedModel,
MultiScaleHyperbolicCombinedModel) and evaluates phase recognition
using K-Nearest Neighbors (k=20).

Usage:
    python eval_knn_phase.py \
        --checkpoint  output/cholec80-hyperbolic-meru/..._checkpoint.bin \
        --model_type  ViT-B_16 \
        --model_variant hyperbolic \
        --train_root       /path/to/cholec80/frames/training_set \
        --test_root        /path/to/cholec80/frames/test_set \
        --train_phase_root /path/to/cholec80/phase_annotations/training_set_1fps \
        --test_phase_root  /path/to/cholec80/phase_annotations/test_set_1fps \
        --img_size 224 \
        --k 20

Phase annotation files are expected as:
    train_phase_root/videoXX-phase.txt   (or XX-phase.txt, etc.)
    test_phase_root/videoXX-phase.txt
with tab-separated columns: Frame\tPhase

The frame filenames in each video folder should be zero-padded integers
(e.g., 000000.jpg, 000001.jpg, ...) matching the 'Frame' column.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.modeling import CONFIGS, VisionTransformer

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# Phase-labelled Dataset
# ═══════════════════════════════════════════════

# Cholec80 canonical phase names (7 phases)
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
PHASE2IDX = {name: idx for idx, name in enumerate(PHASE_NAMES)}


def parse_phase_file(phase_path: str) -> Dict[int, int]:
    """
    Parse a cholec80 phase annotation file.

    Expected format (tab-separated, first line is header):
        Frame\tPhase
        0\tPreparation
        25\tCalotTriangleDissection
        ...

    Also handles space-separated and headerless files.

    Returns:
        dict mapping frame_index -> phase_label (int)
    """
    frame2phase: Dict[int, int] = {}

    with open(phase_path, "r") as f:
        lines = f.readlines()

    # Skip header if present
    start = 0
    first = lines[0].strip()
    if first.startswith("Frame") or not first[0].isdigit():
        start = 1

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\t\s]+", line)
        if len(parts) < 2:
            continue
        frame_idx = int(parts[0])
        phase_name = parts[1]

        if phase_name in PHASE2IDX:
            frame2phase[frame_idx] = PHASE2IDX[phase_name]
        else:
            # Try numeric phase label directly
            try:
                frame2phase[frame_idx] = int(phase_name)
            except ValueError:
                logger.warning(f"Unknown phase '{phase_name}' at frame {frame_idx}")

    return frame2phase


class Cholec80PhaseDataset(Dataset):
    """
    Dataset that loads individual frames with their phase labels.

    Directory structure:
        frame_root/
            01/ (or video01, etc.)
                000000.jpg
                000001.jpg
                ...
        phase_root/
            video01-phase.txt
            video02-phase.txt
            ...

    Each sample: (image_tensor, phase_label)
    """

    def __init__(
        self,
        frame_root: str,
        phase_root: str,
        img_size: int = 224,
        transform=None,
        subsample: int = 1,
    ):
        """
        Args:
            frame_root: path to folder containing per-video frame directories
            phase_root: path to folder containing phase annotation txt files
            img_size: image resolution
            transform: optional transform (default: resize + normalize)
            subsample: take every N-th frame (1 = all frames)
        """
        super().__init__()
        self.frame_root = frame_root
        self.phase_root = phase_root
        self.subsample = subsample

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples: List[Tuple[str, int]] = []  # (frame_path, phase_label)
        self._build_samples()

        logger.info(
            f"[Cholec80PhaseDataset] root={frame_root}, "
            f"samples={len(self.samples)}, subsample={subsample}"
        )

    def _find_phase_file(self, vid_name: str) -> Optional[str]:
        """
        Try multiple naming conventions for phase annotation files:
            video01-phase.txt, 01-phase.txt, video01_phase.txt, etc.
        """
        candidates = [
            f"video{vid_name}-phase.txt",
            f"{vid_name}-phase.txt",
            f"video{vid_name}_phase.txt",
            f"{vid_name}_phase.txt",
            f"video{vid_name}-tool.txt",  # sometimes same folder
        ]

        # Also try zero-padded variations
        try:
            vid_num = int(re.sub(r"[^0-9]", "", vid_name))
            for fmt in [
                f"video{vid_num:02d}-phase.txt",
                f"video{vid_num}-phase.txt",
                f"{vid_num:02d}-phase.txt",
            ]:
                if fmt not in candidates:
                    candidates.append(fmt)
        except ValueError:
            pass

        for c in candidates:
            path = os.path.join(self.phase_root, c)
            if os.path.isfile(path):
                return path
        return None

    def _build_samples(self):
        frame_pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

        video_dirs = sorted([
            d for d in os.listdir(self.frame_root)
            if os.path.isdir(os.path.join(self.frame_root, d))
        ])

        for vid_name in video_dirs:
            phase_file = self._find_phase_file(vid_name)
            if phase_file is None:
                logger.warning(
                    f"No phase annotation found for video '{vid_name}', skipping."
                )
                continue

            frame2phase = parse_phase_file(phase_file)
            if not frame2phase:
                logger.warning(
                    f"Empty phase annotations for video '{vid_name}', skipping."
                )
                continue

            vid_path = os.path.join(self.frame_root, vid_name)
            frames = []
            for fname in os.listdir(vid_path):
                m = frame_pattern.match(fname)
                if m:
                    frame_idx = int(m.group(1))
                    frame_path = os.path.join(vid_path, fname)
                    frames.append((frame_idx, frame_path))

            frames.sort(key=lambda x: x[0])

            # Match frames to phase labels
            count = 0
            for frame_idx, frame_path in frames:
                if frame_idx in frame2phase:
                    count += 1
                    if self.subsample > 1 and count % self.subsample != 0:
                        continue
                    self.samples.append((frame_path, frame2phase[frame_idx]))

        logger.info(
            f"Built {len(self.samples)} samples from "
            f"{len(video_dirs)} video directories."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════
# Feature Extractor
# ═══════════════════════════════════════════════

class ViTFeatureExtractor(nn.Module):
    """
    Wraps a trained model and extracts CLS token features from the ViT backbone.

    Supports all model variants:
        - 'euclidean'    : TemporalViT
        - 'hyperbolic'   : HyperbolicTemporalViT
        - 'combined'     : HyperbolicCombinedModel (entail + PL)
        - 'mat'          : MultiScaleHyperbolicCombinedModel
        - 'backbone_only': raw VisionTransformer
    """

    def __init__(self, config, model_variant: str = "hyperbolic",
                 img_size: int = 224, embed_dim: int = 128,
                 curv_init: float = 1.0, learn_curv: bool = True,
                 score_n_layers: int = 2, score_n_heads: int = 4,
                 score_mlp_ratio: float = 4.0, score_dropout: float = 0.1,
                 pre_split_n_layers: int = 2, pre_split_n_heads: int = 4,
                 pre_split_mlp_ratio: float = 4.0, pre_split_dropout: float = 0.1,
                 scale_weight_temp: float = 1.0):
        super().__init__()
        self.model_variant = model_variant

        if model_variant == "euclidean":
            from models.temporal_vit import TemporalViT
            self.model = TemporalViT(
                config, img_size=img_size, zero_head=True,
            )
            self._get_backbone = lambda: self.model.backbone

        elif model_variant == "hyperbolic":
            from models.temporal_vit import HyperbolicTemporalViT
            self.model = HyperbolicTemporalViT(
                config, img_size=img_size, embed_dim=embed_dim,
                curv_init=curv_init, learn_curv=learn_curv, zero_head=True,
            )
            self._get_backbone = lambda: self.model.backbone

        elif model_variant == "combined":
            from train_hyperbolic_entail_and_pl import HyperbolicCombinedModel
            self.model = HyperbolicCombinedModel(
                config, img_size=img_size, embed_dim=embed_dim,
                curv_init=curv_init, learn_curv=learn_curv,
                score_n_layers=score_n_layers, score_n_heads=score_n_heads,
                score_mlp_ratio=score_mlp_ratio, score_dropout=score_dropout,
                zero_head=True,
            )
            self._get_backbone = lambda: self.model.encoder.backbone

        elif model_variant == "mat":
            from train_hyperbolic_entail_and_pl_mat import MultiScaleHyperbolicCombinedModel
            self.model = MultiScaleHyperbolicCombinedModel(
                config, img_size=img_size, embed_dim=embed_dim,
                curv_init=curv_init, learn_curv=learn_curv,
                pre_split_n_layers=pre_split_n_layers,
                pre_split_n_heads=pre_split_n_heads,
                pre_split_mlp_ratio=pre_split_mlp_ratio,
                pre_split_dropout=pre_split_dropout,
                score_n_layers=score_n_layers, score_n_heads=score_n_heads,
                score_mlp_ratio=score_mlp_ratio, score_dropout=score_dropout,
                scale_weight_temp=scale_weight_temp,
                zero_head=True,
            )
            self._get_backbone = lambda: self.model.encoder.backbone

        elif model_variant == "backbone_only":
            self.model = VisionTransformer(
                config, img_size=img_size, num_classes=1, zero_head=True,
            )
            self._get_backbone = lambda: self.model

        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")

    def load_checkpoint(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {path}")

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features from ViT backbone.
        Args:
            x: (B, 3, H, W) single frames
        Returns:
            (B, hidden_size) CLS token features
        """
        backbone = self._get_backbone()
        encoded, _ = backbone.transformer(x)
        cls_features = encoded[:, 0]  # (B, hidden_size)
        return cls_features


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

@torch.no_grad()
def extract_all_features(
    extractor: ViTFeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from entire dataset."""
    extractor.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = extractor.extract_features(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 20,
) -> dict:
    """
    Run KNN classification and compute metrics.

    Returns:
        dict with accuracy, macro_f1, weighted_f1, per_class_report
    """
    logger.info(f"Fitting KNN with k={k}...")
    logger.info(f"  Train: {train_features.shape[0]} samples, "
                f"Test: {test_features.shape[0]} samples")
    logger.info(f"  Feature dim: {train_features.shape[1]}")

    # L2 normalize features (common practice for KNN on neural features)
    train_norm = train_features / (
        np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8
    )
    test_norm = test_features / (
        np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8
    )

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(train_norm, train_labels)

    pred_labels = knn.predict(test_norm)

    acc = accuracy_score(test_labels, pred_labels)
    macro_f1 = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_labels, pred_labels, average="weighted", zero_division=0)

    # Per-class report
    present_labels = sorted(set(test_labels.tolist()) | set(pred_labels.tolist()))
    target_names = [
        PHASE_NAMES[i] if i < len(PHASE_NAMES) else f"Phase_{i}"
        for i in present_labels
    ]
    report = classification_report(
        test_labels, pred_labels,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "predictions": pred_labels,
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KNN Phase Recognition Evaluation for trained ViT models"
    )

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.bin)")
    parser.add_argument("--model_type", type=str, default="ViT-B_16",
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"])
    parser.add_argument("--model_variant", type=str, default="hyperbolic",
                        choices=["euclidean", "hyperbolic", "combined", "mat",
                                 "backbone_only"],
                        help="Which model architecture was used for training.")
    parser.add_argument("--img_size", type=int, default=224)

    # Hyperbolic params (needed for model construction)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--curv_init", type=float, default=1.0)
    parser.add_argument("--learn_curv", action="store_true", default=True)
    parser.add_argument("--no_learn_curv", dest="learn_curv", action="store_false")

    # Score head params (for combined / mat variants)
    parser.add_argument("--score_n_layers", type=int, default=2)
    parser.add_argument("--score_n_heads", type=int, default=4)
    parser.add_argument("--score_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--score_dropout", type=float, default=0.1)

    # MAT-specific params
    parser.add_argument("--pre_split_n_layers", type=int, default=2)
    parser.add_argument("--pre_split_n_heads", type=int, default=4)
    parser.add_argument("--pre_split_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--pre_split_dropout", type=float, default=0.1)
    parser.add_argument("--scale_weight_temp", type=float, default=1.0)

    # Data
    parser.add_argument("--train_root", type=str, required=True,
                        help="Path to training video frame directories.")
    parser.add_argument("--test_root", type=str, required=True,
                        help="Path to test video frame directories.")
    parser.add_argument("--train_phase_root", type=str, required=True,
                        help="Path to training phase annotation .txt files "
                             "(e.g., .../phase_annotations/training_set_1fps).")
    parser.add_argument("--test_phase_root", type=str, required=True,
                        help="Path to test phase annotation .txt files "
                             "(e.g., .../phase_annotations/test_set_1fps).")
    parser.add_argument("--subsample", type=int, default=25,
                        help="Take every N-th frame to reduce computation "
                             "(1 = all frames, 25 = 1fps if 25fps video).")

    # KNN
    parser.add_argument("--k", type=int, default=20,
                        help="Number of neighbors for KNN.")

    # Misc
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save results as .txt")

    args = parser.parse_args()

    # ── Logging ──
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model & load checkpoint ──
    config = CONFIGS[args.model_type]

    extractor = ViTFeatureExtractor(
        config=config,
        model_variant=args.model_variant,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        curv_init=args.curv_init,
        learn_curv=args.learn_curv,
        score_n_layers=args.score_n_layers,
        score_n_heads=args.score_n_heads,
        score_mlp_ratio=args.score_mlp_ratio,
        score_dropout=args.score_dropout,
        pre_split_n_layers=args.pre_split_n_layers,
        pre_split_n_heads=args.pre_split_n_heads,
        pre_split_mlp_ratio=args.pre_split_mlp_ratio,
        pre_split_dropout=args.pre_split_dropout,
        scale_weight_temp=args.scale_weight_temp,
    )
    extractor.load_checkpoint(args.checkpoint)
    extractor.to(device)
    extractor.eval()

    # ── Datasets ──
    logger.info("Building train dataset...")
    train_dataset = Cholec80PhaseDataset(
        frame_root=args.train_root,
        phase_root=args.train_phase_root,
        img_size=args.img_size,
        subsample=args.subsample,
    )

    logger.info("Building test dataset...")
    test_dataset = Cholec80PhaseDataset(
        frame_root=args.test_root,
        phase_root=args.test_phase_root,
        img_size=args.img_size,
        subsample=args.subsample,
    )

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Train or test dataset is empty. Check paths and annotations.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(train_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Extract features ──
    logger.info("Extracting train features...")
    train_features, train_labels = extract_all_features(extractor, train_loader, device)

    logger.info("Extracting test features...")
    test_features, test_labels = extract_all_features(extractor, test_loader, device)

    # ── Label distribution ──
    logger.info("Train label distribution:")
    for idx, name in enumerate(PHASE_NAMES):
        count = (train_labels == idx).sum()
        if count > 0:
            logger.info(f"  {name}: {count}")

    logger.info("Test label distribution:")
    for idx, name in enumerate(PHASE_NAMES):
        count = (test_labels == idx).sum()
        if count > 0:
            logger.info(f"  {name}: {count}")

    # ── KNN Evaluation ──
    results = evaluate_knn(
        train_features, train_labels,
        test_features, test_labels,
        k=args.k,
    )

    # ── Print results ──
    print("\n" + "=" * 60)
    print(f"KNN Phase Recognition Results (k={args.k})")
    print("=" * 60)
    print(f"  Accuracy:     {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1:     {results['macro_f1']:.4f}")
    print(f"  Weighted F1:  {results['weighted_f1']:.4f}")
    print("\nPer-class Classification Report:")
    print(results["report"])

    # ── Save results ──
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write(f"KNN Phase Recognition Results (k={args.k})\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Model: {args.model_type} ({args.model_variant})\n")
            f.write(f"Subsample: {args.subsample}\n")
            f.write(f"Train samples: {len(train_features)}\n")
            f.write(f"Test samples: {len(test_features)}\n")
            f.write(f"Feature dim: {train_features.shape[1]}\n\n")
            f.write(f"Accuracy:    {results['accuracy']:.4f}\n")
            f.write(f"Macro F1:    {results['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results["report"])
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()