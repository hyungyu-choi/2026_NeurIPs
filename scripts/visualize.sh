#!/bin/bash

# ========================
# Lorentz Height-based Embedding Visualization
# ========================
# Euclidean vs Hyperbolic 임베딩을 비교합니다.
#
# Hyperbolic 모델: Lorentz hyperboloid 원점까지의 측지 거리(height)로 계층 구조 시각화
# Euclidean 모델: L2 norm을 height 대용으로 사용
#
# 생성되는 그래프:
#   - *_height_profile.png  : Video progress vs Height (phase별 색상)
#   - *_phase_boxplot.png   : Phase별 height 분포 (boxplot)
#   - *_2d_scatter.png      : PCA 2D 투영 (phase 색상 / height 색상)
#   - *_trajectories.png    : 개별 비디오의 height 궤적
#   - comparison.png        : 모델 간 나란히 비교
# ========================

# ── Dataset paths ──
TEST_ROOT="../code/Dataset/cholec80/frames/extract_1fps/test_set"
PHASE_ROOT="../code/Dataset/cholec80/phase_annotations/test_set_1fps"

# ── Output ──
OUTPUT_DIR="visualizations"
IMG_SIZE=224
BATCH_SIZE=64
SAMPLE_STEP=30           # 매 N번째 프레임만 사용
MAX_FRAMES_PER_VIDEO=300

# ── GPU ──
GPU_ID="1"

# ════════════════════════════════════════════
# Checkpoint 설정 — 학습한 모델 경로를 수정하세요
# 학습하지 않은 모델은 해당 변수를 비워두세요 (CKPT="")
# ════════════════════════════════════════════

# # 1) Euclidean (TemporalViT + Plackett-Luce)
# EUCLIDEAN_CKPT="output/cholec80-temporal-ordering/cholec80-temporal-ordering_step50000_checkpoint.bin"
# EUCLIDEAN_MODEL_TYPE="ViT-B_16"

# # 2) Hyperbolic (MERU-style entailment only)
# HYPERBOLIC_CKPT="output/cholec80-hyperbolic-meru/cholec80-hyperbolic-meru_step50000_checkpoint.bin"
# HYPERBOLIC_MODEL_TYPE="ViT-B_16"
# HYPERBOLIC_EMBED_DIM=128

# 3) Hyperbolic + Plackett-Luce (combined)
HYPER_PL_CKPT="output/cholec80-hyperbolic-entail-pl-mat/cholec80-hyperbolic-entail-pl-mat_step37000_checkpoint.bin"
HYPER_PL_MODEL_TYPE="ViT-B_16"
HYPER_PL_EMBED_DIM=128

# ════════════════════════════════════════════
# Build command
# ════════════════════════════════════════════

CHECKPOINTS=""
MODEL_TYPES=""
ARCH_TYPES=""
EMBED_DIMS=""

if [ -n "$EUCLIDEAN_CKPT" ] && [ -f "$EUCLIDEAN_CKPT" ]; then
    CHECKPOINTS="$CHECKPOINTS euclidean:$EUCLIDEAN_CKPT"
    MODEL_TYPES="$MODEL_TYPES euclidean:$EUCLIDEAN_MODEL_TYPE"
    ARCH_TYPES="$ARCH_TYPES euclidean:euclidean"
    echo "[✓] Euclidean: $EUCLIDEAN_CKPT"
else
    echo "[✗] Euclidean not found: $EUCLIDEAN_CKPT"
fi

if [ -n "$HYPERBOLIC_CKPT" ] && [ -f "$HYPERBOLIC_CKPT" ]; then
    CHECKPOINTS="$CHECKPOINTS hyperbolic:$HYPERBOLIC_CKPT"
    MODEL_TYPES="$MODEL_TYPES hyperbolic:$HYPERBOLIC_MODEL_TYPE"
    ARCH_TYPES="$ARCH_TYPES hyperbolic:hyperbolic"
    EMBED_DIMS="$EMBED_DIMS hyperbolic:$HYPERBOLIC_EMBED_DIM"
    echo "[✓] Hyperbolic: $HYPERBOLIC_CKPT"
else
    echo "[✗] Hyperbolic not found: $HYPERBOLIC_CKPT"
fi

if [ -n "$HYPER_PL_CKPT" ] && [ -f "$HYPER_PL_CKPT" ]; then
    CHECKPOINTS="$CHECKPOINTS hyper_pl:$HYPER_PL_CKPT"
    MODEL_TYPES="$MODEL_TYPES hyper_pl:$HYPER_PL_MODEL_TYPE"
    ARCH_TYPES="$ARCH_TYPES hyper_pl:hyperbolic_combined"
    EMBED_DIMS="$EMBED_DIMS hyper_pl:$HYPER_PL_EMBED_DIM"
    echo "[✓] Hyperbolic+PL: $HYPER_PL_CKPT"
else
    echo "[✗] Hyperbolic+PL not found: $HYPER_PL_CKPT"
fi

if [ -z "$CHECKPOINTS" ]; then
    echo ""; echo "ERROR: No checkpoints found!"; exit 1
fi

echo ""
echo "=============================="
echo "Lorentz Height Visualization"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "=============================="

CMD="python3 visualize_embeddings.py \
    --test_root $TEST_ROOT \
    --phase_root $PHASE_ROOT \
    --checkpoints $CHECKPOINTS \
    --model_types $MODEL_TYPES \
    --arch_types $ARCH_TYPES \
    --output_dir $OUTPUT_DIR \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --sample_step $SAMPLE_STEP \
    --max_frames_per_video $MAX_FRAMES_PER_VIDEO"

[ -n "$EMBED_DIMS" ] && CMD="$CMD --embed_dims $EMBED_DIMS"

echo ""; echo "$CMD"; echo ""
CUDA_VISIBLE_DEVICES=$GPU_ID $CMD

echo ""
echo "Done! Results → $OUTPUT_DIR/"