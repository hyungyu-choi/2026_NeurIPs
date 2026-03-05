# coding=utf-8
"""
Lorentz-native embedding visualization for temporal ordering models.

Euclidean vs Hyperbolic 비교를 위해 각 공간의 특성에 맞는 시각화를 생성합니다.

핵심 아이디어:
  - Lorentz 임베딩의 계층 정보 = 원점까지의 측지 거리 (height)
  - 시간적으로 이른 프레임은 origin 가까이 (낮은 height), 늦은 프레임은 먼 곳 (높은 height)
  - Phase별로 height가 잘 분리되면 → 계층 구조를 잘 학습한 것

생성되는 시각화:
  1) Height Profile: 프레임의 시간 위치(%) vs height, phase별 색상
  2) Phase별 Height Boxplot: 각 phase의 height 분포 비교
  3) 2D Embedding Scatter: PCA 2D 투영 + height를 점 크기로 표현
  4) Per-video Height Trajectory: 개별 비디오의 height 변화 궤적
  5) Euclidean vs Hyperbolic 나란히 비교
"""
from __future__ import annotations

import argparse
import os
import re
import math
import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# ── Project imports ──
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.modeling import CONFIGS
from models.temporal_vit import (
    TemporalViT,
    HyperbolicTemporalViT,
)
from models import lorentz_ops as L

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# Phase colors (cholec80 surgical phases, temporal order)
# ══════════════════════════════════════════════

PHASE_ORDER = [
    'Preparation',
    'CalotTriangleDissection',
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderPackaging',
    'CleaningCoagulation',
    'GallbladderRetraction',
]

PHASE_COLORS = {
    'Preparation':             '#e6194b',
    'CalotTriangleDissection': '#3cb44b',
    'ClippingCutting':         '#4363d8',
    'GallbladderDissection':   '#f58231',
    'GallbladderPackaging':    '#911eb4',
    'CleaningCoagulation':     '#42d4f4',
    'GallbladderRetraction':   '#f032e6',
}

_EXTRA_COLORS = [
    '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075',
]


def get_phase_color(phase_name: str, _cache={}, _idx=[0]) -> str:
    if phase_name in PHASE_COLORS:
        return PHASE_COLORS[phase_name]
    if phase_name not in _cache:
        _cache[phase_name] = _EXTRA_COLORS[_idx[0] % len(_EXTRA_COLORS)]
        _idx[0] += 1
    return _cache[phase_name]


def phase_sort_key(phase_name: str) -> int:
    if phase_name in PHASE_ORDER:
        return PHASE_ORDER.index(phase_name)
    return len(PHASE_ORDER)


# ══════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════

def load_phase_annotations(phase_root: str) -> Dict[str, Dict[int, str]]:
    annotations = {}
    for fname in sorted(os.listdir(phase_root)):
        if not fname.endswith('-phase.txt'):
            continue
        vid_num = re.search(r'(\d+)', fname)
        if vid_num is None:
            continue
        vid_key = vid_num.group(1)

        filepath = os.path.join(phase_root, fname)
        frame2phase = {}
        with open(filepath, 'r') as f:
            header = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if header:
                    header = False
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frame2phase[int(parts[0])] = parts[1]
        annotations[vid_key] = frame2phase
    logger.info(f"Loaded phase annotations for {len(annotations)} videos")
    return annotations


def scan_test_frames(
    test_root: str,
    annotations: Dict[str, Dict[int, str]],
    max_frames_per_video: int = 300,
    sample_step: int = 10,
) -> Tuple[List[str], List[str], List[int], List[float]]:
    """
    Returns:
        frame_paths, frame_phases, frame_vid_ids, frame_progress (0~1 within video)
    """
    frame_paths, frame_phases, frame_vid_ids, frame_progress = [], [], [], []

    video_dirs = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])
    frame_pattern = re.compile(r'(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)

    for vid_dir in video_dirs:
        vid_num = re.search(r'(\d+)', vid_dir)
        if vid_num is None:
            continue
        vid_key = vid_num.group(1)
        if vid_key not in annotations:
            continue

        vid_path = os.path.join(test_root, vid_dir)
        frames = []
        for fname in os.listdir(vid_path):
            m = frame_pattern.match(fname)
            if m:
                frames.append((int(m.group(1)), os.path.join(vid_path, fname)))
        frames.sort(key=lambda x: x[0])
        if not frames:
            continue

        total_frames = len(frames)
        selected = frames[::sample_step][:max_frames_per_video]
        phase_map = annotations[vid_key]

        for fidx, fpath in selected:
            if fidx in phase_map:
                frame_paths.append(fpath)
                frame_phases.append(phase_map[fidx])
                frame_vid_ids.append(int(vid_key))
                frame_progress.append(fidx / max(total_frames - 1, 1))

    logger.info(f"Selected {len(frame_paths)} frames from {len(video_dirs)} videos")
    return frame_paths, frame_phases, frame_vid_ids, frame_progress


# ══════════════════════════════════════════════
# Model building & embedding extraction
# ══════════════════════════════════════════════

def build_euclidean_model(config, img_size, checkpoint_path, device):
    model = TemporalViT(
        config, img_size=img_size, pretrained_weights=None,
        hidden_mul=0.5, max_len=64, n_layers=6, n_head=8,
        dropout=0.1, zero_head=True, vis=False,
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def build_hyperbolic_model(config, img_size, checkpoint_path, device,
                           embed_dim=128, curv_init=1.0):
    model = HyperbolicTemporalViT(
        config, img_size=img_size, pretrained_weights=None,
        embed_dim=embed_dim, curv_init=curv_init, learn_curv=True,
        zero_head=True, vis=False,
    )
    state = torch.load(checkpoint_path, map_location='cpu')

    # Handle HyperbolicCombinedModel checkpoints (encoder. prefix)
    if any(k.startswith('encoder.') for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            if k.startswith('encoder.'):
                new_state[k[len('encoder.'):]] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_euclidean_embeddings(
    model, frame_paths, device, img_size=224, batch_size=64
) -> np.ndarray:
    """CLS token embeddings from ViT backbone."""
    tx = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_embs = []
    for i in range(0, len(frame_paths), batch_size):
        batch = [tx(Image.open(p).convert('RGB')) for p in frame_paths[i:i+batch_size]]
        imgs = torch.stack(batch).to(device)
        encoded, _ = model.backbone.transformer(imgs)
        all_embs.append(encoded[:, 0].cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Euclidean: {i}/{len(frame_paths)}")
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def extract_hyperbolic_embeddings(
    model, frame_paths, device, img_size=224, batch_size=64
) -> Tuple[np.ndarray, float]:
    """Lorentz space-component embeddings + curvature."""
    tx = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    curv_val = model.curvature.item()
    all_embs = []

    for i in range(0, len(frame_paths), batch_size):
        batch = [tx(Image.open(p).convert('RGB')) for p in frame_paths[i:i+batch_size]]
        imgs = torch.stack(batch).to(device)
        encoded, _ = model.backbone.transformer(imgs)
        cls = encoded[:, 0]
        h_space = model.lorentz_proj(cls)
        h_space = torch.nan_to_num(h_space, nan=0.0, posinf=0.0, neginf=0.0)
        all_embs.append(h_space.cpu().float().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Hyperbolic: {i}/{len(frame_paths)}")

    embeddings = np.concatenate(all_embs, axis=0)
    logger.info(f"  Stats: shape={embeddings.shape}, "
                f"norm_mean={np.linalg.norm(embeddings, axis=-1).mean():.4f}, "
                f"norm_max={np.linalg.norm(embeddings, axis=-1).max():.4f}, "
                f"NaN={np.isnan(embeddings).any()}")
    return embeddings, curv_val


def compute_lorentz_heights(x_space: np.ndarray, curv: float) -> np.ndarray:
    """Geodesic distance from origin on the hyperboloid = hierarchy height."""
    x_space = np.nan_to_num(x_space, nan=0.0, posinf=0.0, neginf=0.0)
    sq_norms = np.sum(x_space ** 2, axis=-1)
    x_time = np.sqrt(np.clip(1.0 / curv + sq_norms, 1e-8, None))
    rc_x_time = np.sqrt(curv) * x_time
    rc_x_time = np.clip(rc_x_time, 1.0 + 1e-8, None)
    dist = np.arccosh(rc_x_time) / np.sqrt(curv)
    return np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)


def compute_euclidean_norms(embeddings: np.ndarray) -> np.ndarray:
    """L2 norm as proxy for distance from origin in Euclidean space."""
    return np.linalg.norm(embeddings, axis=-1)


# ══════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════

def _make_legend_handles(phases):
    sorted_phases = sorted(set(phases), key=phase_sort_key)
    return [mpatches.Patch(color=get_phase_color(p), label=p) for p in sorted_phases]


# ── 1) Height Profile ──

def plot_height_profile(heights, progress, phases, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_phases = sorted(set(phases), key=phase_sort_key)
    for phase in sorted_phases:
        mask = np.array([p == phase for p in phases])
        ax.scatter(progress[mask], heights[mask],
                   c=get_phase_color(phase), s=4, alpha=0.4,
                   label=phase, edgecolors='none', rasterized=True)
    ax.set_xlabel('Video Progress (0=start, 1=end)', fontsize=12)
    ax.set_ylabel('Height (geodesic dist. to origin)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(handles=_make_legend_handles(phases),
              loc='upper left', fontsize=8, framealpha=0.9,
              bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── 2) Phase Height Boxplot ──

def plot_phase_boxplot(heights, phases, title, save_path):
    sorted_phases = sorted(set(phases), key=phase_sort_key)
    data_by_phase, colors, labels = [], [], []
    for phase in sorted_phases:
        mask = np.array([p == phase for p in phases])
        data_by_phase.append(heights[mask])
        colors.append(get_phase_color(phase))
        labels.append(phase)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_by_phase, patch_artist=True, labels=labels,
                     showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Strip-plot overlay
    for i, (phase, d) in enumerate(zip(sorted_phases, data_by_phase)):
        jitter = np.random.normal(0, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d,
                   c=get_phase_color(phase), s=2, alpha=0.15,
                   edgecolors='none', rasterized=True)

    ax.set_ylabel('Height (geodesic dist. to origin)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', alpha=0.2)

    # Temporal order arrow
    ymax = ax.get_ylim()[1]
    ax.annotate('', xy=(len(sorted_phases) + 0.3, ymax * 0.95),
                xytext=(0.7, ymax * 0.95),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(len(sorted_phases) / 2 + 0.5, ymax * 0.98,
            'Temporal order →', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── 3) 2D Embedding Scatter ──

def plot_2d_scatter(embeddings, heights, phases, title, save_path):
    clean = np.nan_to_num(embeddings, nan=0.0)
    if clean.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(clean)
        vr = pca.explained_variance_ratio_
        xl, yl = f'PC1 ({vr[0]:.1%})', f'PC2 ({vr[1]:.1%})'
    else:
        coords = clean
        xl, yl = 'Dim 1', 'Dim 2'

    sorted_phases = sorted(set(phases), key=phase_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: phase-colored
    ax = axes[0]
    for phase in sorted_phases:
        mask = np.array([p == phase for p in phases])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=get_phase_color(phase), s=5, alpha=0.4,
                   edgecolors='none', rasterized=True)
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f'{title}\n(colored by phase)', fontsize=12, fontweight='bold')
    ax.legend(handles=_make_legend_handles(phases), fontsize=7,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # Right: height-colored
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=heights, cmap='viridis', s=5, alpha=0.5,
                    edgecolors='none', rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Height (dist. to origin)', fontsize=10)
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f'{title}\n(colored by height)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── 4) Per-video Trajectory ──

def plot_video_trajectories(heights, progress, phases, vid_ids, title, save_path,
                            max_videos=6):
    unique_vids = sorted(set(vid_ids))
    selected = unique_vids[:max_videos]
    n = len(selected)
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, vid_id in enumerate(selected):
        ax = axes[idx]
        mask = np.array([v == vid_id for v in vid_ids])
        vp = progress[mask]; vh = heights[mask]
        vph = [phases[i] for i, m in enumerate(mask) if m]
        si = np.argsort(vp); vp = vp[si]; vh = vh[si]
        vph = [vph[i] for i in si]

        ax.plot(vp, vh, color='gray', alpha=0.3, linewidth=0.8)
        for phase in sorted(set(vph), key=phase_sort_key):
            pmask = np.array([p == phase for p in vph])
            ax.scatter(vp[pmask], vh[pmask], c=get_phase_color(phase),
                       s=12, alpha=0.7, edgecolors='none', label=phase)
        ax.set_xlabel('Progress', fontsize=9)
        ax.set_ylabel('Height', fontsize=9)
        ax.set_title(f'Video {vid_id}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles=_make_legend_handles(phases), loc='lower center',
               ncol=min(len(set(phases)), 4), fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── 5) Side-by-side Comparison ──

def plot_comparison_figure(results, phases, progress, save_path):
    n = len(results)
    fig = plt.figure(figsize=(7*n, 18))
    gs = gridspec.GridSpec(3, n, hspace=0.35, wspace=0.3)
    sorted_phases = sorted(set(phases), key=phase_sort_key)

    for col, res in enumerate(results):
        name, heights, embs = res['name'], res['heights'], res['embeddings']

        # Row 0: Height Profile
        ax = fig.add_subplot(gs[0, col])
        for phase in sorted_phases:
            mask = np.array([p == phase for p in phases])
            ax.scatter(progress[mask], heights[mask],
                       c=get_phase_color(phase), s=3, alpha=0.35,
                       edgecolors='none', rasterized=True)
        ax.set_xlabel('Video Progress', fontsize=10)
        ax.set_ylabel('Height', fontsize=10)
        ax.set_title(f'{name}\nHeight Profile', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Row 1: Boxplot
        ax = fig.add_subplot(gs[1, col])
        data_bp, colors_bp, labels_bp = [], [], []
        for phase in sorted_phases:
            mask = np.array([p == phase for p in phases])
            if mask.sum() > 0:
                data_bp.append(heights[mask])
                colors_bp.append(get_phase_color(phase))
                labels_bp.append(phase[:12])
        bp = ax.boxplot(data_bp, patch_artist=True, labels=labels_bp,
                         showfliers=False, widths=0.6)
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for m in bp['medians']:
            m.set_color('black'); m.set_linewidth(2)
        ax.set_ylabel('Height', fontsize=10)
        ax.set_title('Phase Height Distribution', fontsize=10)
        ax.tick_params(axis='x', rotation=35, labelsize=7)
        ax.grid(True, axis='y', alpha=0.2)

        # Row 2: 2D PCA scatter
        ax = fig.add_subplot(gs[2, col])
        clean = np.nan_to_num(embs, nan=0.0)
        coords = PCA(n_components=2).fit_transform(clean) if clean.shape[1] > 2 else clean
        for phase in sorted_phases:
            mask = np.array([p == phase for p in phases])
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=get_phase_color(phase), s=3, alpha=0.35,
                       edgecolors='none', rasterized=True)
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_title('2D Projection (PCA)', fontsize=10)
        ax.grid(True, alpha=0.15)

    handles = _make_legend_handles(phases)
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(sorted_phases), 4), fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved comparison: {save_path}")


# ── 6) Metrics ──

def compute_metrics(heights, phases, embeddings):
    sorted_phases = sorted(set(phases), key=phase_sort_key)
    phase_labels = np.array([phase_sort_key(p) for p in phases])

    phase_stats = {}
    phase_medians = []
    for phase in sorted_phases:
        mask = np.array([p == phase for p in phases])
        h = heights[mask]
        phase_stats[phase] = {
            'mean': float(np.mean(h)), 'median': float(np.median(h)),
            'std': float(np.std(h)), 'count': int(mask.sum()),
        }
        phase_medians.append(np.median(h))

    if len(phase_medians) > 2:
        rank_corr, rank_p = spearmanr(range(len(phase_medians)), phase_medians)
    else:
        rank_corr, rank_p = float('nan'), float('nan')

    clean = np.nan_to_num(embeddings, nan=0.0)
    coords = PCA(n_components=2).fit_transform(clean) if clean.shape[1] > 2 else clean
    unique_labels = np.unique(phase_labels)
    if len(unique_labels) > 1 and len(coords) > 20:
        try:
            sil = silhouette_score(coords, phase_labels,
                                   sample_size=min(5000, len(phase_labels)))
        except Exception:
            sil = float('nan')
    else:
        sil = float('nan')

    return {
        'phase_stats': phase_stats,
        'height_rank_correlation': rank_corr,
        'height_rank_p_value': rank_p,
        'silhouette_score': sil,
    }


def print_metrics(name, metrics):
    logger.info(f"\n{'─'*50}")
    logger.info(f"  Metrics: {name}")
    logger.info(f"{'─'*50}")
    logger.info(f"  Silhouette score:       {metrics['silhouette_score']:.4f}")
    logger.info(f"  Height-phase rank corr: {metrics['height_rank_correlation']:.4f} "
                f"(p={metrics['height_rank_p_value']:.4f})")
    logger.info(f"  Per-phase height (median ± std):")
    for phase, s in metrics['phase_stats'].items():
        logger.info(f"    {phase:<30s}: {s['median']:.4f} ± {s['std']:.4f}  (n={s['count']})")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def parse_keyval_list(items):
    d = {}
    for item in items:
        parts = item.split(':', 1)
        if len(parts) == 2:
            d[parts[0]] = parts[1]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--phase_root", type=str, required=True)
    parser.add_argument("--checkpoints", nargs='+', required=True,
                        help="name:path pairs")
    parser.add_argument("--model_types", nargs='+', default=None,
                        help="name:ViT-B_16 pairs")
    parser.add_argument("--arch_types", nargs='+', required=True,
                        help="name:arch (euclidean | hyperbolic | hyperbolic_combined)")
    parser.add_argument("--embed_dims", nargs='+', default=None,
                        help="name:dim pairs (default 128)")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_step", type=int, default=30)
    parser.add_argument("--max_frames_per_video", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt_map = parse_keyval_list(args.checkpoints)
    arch_map = parse_keyval_list(args.arch_types)
    mt_map = parse_keyval_list(args.model_types) if args.model_types else {}
    ed_map = parse_keyval_list(args.embed_dims) if args.embed_dims else {}

    annotations = load_phase_annotations(args.phase_root)
    frame_paths, frame_phases, frame_vid_ids, frame_progress = scan_test_frames(
        args.test_root, annotations,
        max_frames_per_video=args.max_frames_per_video,
        sample_step=args.sample_step,
    )
    if not frame_paths:
        logger.error("No frames found!")
        return

    progress_arr = np.array(frame_progress)

    logger.info("Phase distribution:")
    pc = defaultdict(int)
    for p in frame_phases:
        pc[p] += 1
    for p, c in sorted(pc.items()):
        logger.info(f"  {p}: {c}")

    # ── Process each model ──
    all_results = []

    for name, ckpt_path in ckpt_map.items():
        arch = arch_map.get(name, 'euclidean')
        model_type = mt_map.get(name, 'ViT-B_16')
        embed_dim = int(ed_map.get(name, '128'))
        config = CONFIGS[model_type]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name} (arch={arch})")
        logger.info(f"Checkpoint: {ckpt_path}")
        logger.info(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            continue

        if arch == 'euclidean':
            model = build_euclidean_model(config, args.img_size, ckpt_path, device)
            embeddings = extract_euclidean_embeddings(
                model, frame_paths, device, args.img_size, args.batch_size)
            heights = compute_euclidean_norms(embeddings)
            display_name = f"{name} (Euclidean)"

        elif arch in ('hyperbolic', 'hyperbolic_combined'):
            model = build_hyperbolic_model(
                config, args.img_size, ckpt_path, device, embed_dim=embed_dim)
            embeddings, curv = extract_hyperbolic_embeddings(
                model, frame_paths, device, args.img_size, args.batch_size)
            heights = compute_lorentz_heights(embeddings, curv)
            display_name = f"{name} (Hyperbolic, c={curv:.4f})"
            logger.info(f"  Learned curvature: {curv:.6f}")
        else:
            logger.error(f"Unknown arch: {arch}")
            continue

        result = {'name': display_name, 'heights': heights, 'embeddings': embeddings}
        all_results.append(result)

        prefix = os.path.join(args.output_dir, name)

        plot_height_profile(heights, progress_arr, frame_phases,
                            f'{display_name} — Height Profile',
                            f'{prefix}_height_profile.png')

        plot_phase_boxplot(heights, frame_phases,
                           f'{display_name} — Phase Height Distribution',
                           f'{prefix}_phase_boxplot.png')

        plot_2d_scatter(embeddings, heights, frame_phases,
                        display_name, f'{prefix}_2d_scatter.png')

        plot_video_trajectories(heights, progress_arr, frame_phases, frame_vid_ids,
                                f'{display_name} — Video Trajectories',
                                f'{prefix}_trajectories.png')

        metrics = compute_metrics(heights, frame_phases, embeddings)
        print_metrics(display_name, metrics)

        del model
        torch.cuda.empty_cache()

    if len(all_results) >= 2:
        plot_comparison_figure(all_results, frame_phases, progress_arr,
                               os.path.join(args.output_dir, "comparison.png"))

    logger.info(f"\nAll visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()