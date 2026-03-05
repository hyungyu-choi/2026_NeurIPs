#!/bin/bash

# ========================
# KNN Phase Recognition Evaluation
# ========================
# Extracts CLS token features from trained ViT backbone
# and evaluates phase recognition via KNN (k=20).
# ========================

# ─── Checkpoint ───
CHECKPOINT="output/cholec80-hyperbolic-entail-pl-mat/cholec80-hyperbolic-entail-pl-mat_step10000_checkpoint.bin"
MODEL_TYPE="ViT-B_16"

# Model variant: "euclidean" | "hyperbolic" | "combined" | "mat" | "backbone_only"
MODEL_VARIANT="mat"

# ─── Dataset paths ───
TRAIN_ROOT="../code/Dataset/cholec80/frames/extract_1fps/training_set"
TEST_ROOT="../code/Dataset/cholec80/frames/extract_1fps/test_set"
TRAIN_PHASE_ROOT="../code/Dataset/cholec80/phase_annotations/training_set_1fps"
TEST_PHASE_ROOT="../code/Dataset/cholec80/phase_annotations/test_set_1fps"

IMG_SIZE=224

# 프레임 서브샘플링 (25fps 비디오에서 1fps로 줄이려면 25, 이미 1fps이면 1)
SUBSAMPLE=1

# ─── Hyperbolic params (모델 구조 재현용, 학습 시 설정과 동일하게) ───
EMBED_DIM=128
CURV_INIT=1.0
LEARN_CURV="--learn_curv"

# ─── Score head params (combined / mat variant용) ───
SCORE_N_LAYERS=2
SCORE_N_HEADS=4
SCORE_MLP_RATIO=4.0
SCORE_DROPOUT=0.1

# ─── MAT-specific params ───
PRE_SPLIT_N_LAYERS=2
PRE_SPLIT_N_HEADS=4
PRE_SPLIT_MLP_RATIO=4.0
PRE_SPLIT_DROPOUT=0.1
SCALE_WEIGHT_TEMP=1.0

# ─── KNN ───
K=20

# ─── Misc ───
BATCH_SIZE=128
NUM_WORKERS=4
OUTPUT_FILE="output/eval_results/knn_k${K}_${MODEL_VARIANT}.txt"

# ─── GPU ───
GPU_ID="2"

# ========================
# 실행
# ========================
echo "=============================="
echo "KNN Phase Recognition Evaluation"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model: $MODEL_TYPE ($MODEL_VARIANT)"
echo "  k = $K"
echo "  GPU: $GPU_ID"
echo "=============================="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 eval_knn.py \
    --checkpoint "$CHECKPOINT" \
    --model_type "$MODEL_TYPE" \
    --model_variant "$MODEL_VARIANT" \
    --img_size $IMG_SIZE \
    --embed_dim $EMBED_DIM \
    --curv_init $CURV_INIT \
    $LEARN_CURV \
    --score_n_layers $SCORE_N_LAYERS \
    --score_n_heads $SCORE_N_HEADS \
    --score_mlp_ratio $SCORE_MLP_RATIO \
    --score_dropout $SCORE_DROPOUT \
    --pre_split_n_layers $PRE_SPLIT_N_LAYERS \
    --pre_split_n_heads $PRE_SPLIT_N_HEADS \
    --pre_split_mlp_ratio $PRE_SPLIT_MLP_RATIO \
    --pre_split_dropout $PRE_SPLIT_DROPOUT \
    --scale_weight_temp $SCALE_WEIGHT_TEMP \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --train_phase_root "$TRAIN_PHASE_ROOT" \
    --test_phase_root "$TEST_PHASE_ROOT" \
    --subsample $SUBSAMPLE \
    --k $K \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_file "$OUTPUT_FILE"