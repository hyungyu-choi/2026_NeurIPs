#!/bin/bash

# ========================
# Hyperbolic PL-Only Training
# ========================
# Architecture:
#   ViT → MERUStyleProjection (hyperboloid, dim D) → LorentzScoreHead → PL scores
#   Loss: Plackett-Luce only (no entailment, no multi-scale)
# ========================
NAME="cholec80-hyperbolic-pl-only"
MODEL_TYPE="ViT-B_16"
# PRETRAINED_DIR="checkpoint/ViT-B_16.npz"
OUTPUT_DIR="output"

# Dataset
DATA_ROOT="../code/Dataset/cholec80/frames/extract_1fps/training_set"
# VAL_ROOT="../code/Dataset/cholec80/frames/extract_1fps/val_set"

# Temporal sampling
IMG_SIZE=224
SEQ_LEN=8
MIN_STEP=1
MAX_STEP=20
SAMPLING_MODE="global"

# Hyperbolic space (MERU-style)
EMBED_DIM=128
CURV_INIT=1.0
LEARN_CURV="--learn_curv"

# Plackett-Luce loss
PL_SAMPLE=""                # "--pl_sample" 으로 설정하면 subset sampling 사용
PL_R=4
PL_K=8

# Lorentz Score Head
SCORE_N_LAYERS=2
SCORE_N_HEADS=4
SCORE_MLP_RATIO=4.0
SCORE_DROPOUT=0.1

# Training
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=16
EVAL_EVERY=500

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.05
NUM_STEPS=50000
DECAY_TYPE="cosine"
WARMUP_STEPS=1000
MAX_GRAD_NORM=1.0
SEED=42
GRADIENT_ACCUMULATION_STEPS=1
FP16=""                     # "--fp16" 으로 설정하면 mixed precision 사용

# Wandb
USE_WANDB="--use_wandb"
WANDB_PROJECT="hyperbolic-temporal-vit"
# WANDB_ENTITY=""

# GPU
GPU_IDS="3"
MODE="single"

# ========================
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

COMMON_ARGS="
    --name $NAME \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --img_size $IMG_SIZE \
    --seq_len $SEQ_LEN \
    --min_step $MIN_STEP \
    --max_step $MAX_STEP \
    --sampling_mode $SAMPLING_MODE \
    --embed_dim $EMBED_DIM \
    --curv_init $CURV_INIT \
    --pl_R $PL_R \
    --pl_K $PL_K \
    --score_n_layers $SCORE_N_LAYERS \
    --score_n_heads $SCORE_N_HEADS \
    --score_mlp_ratio $SCORE_MLP_RATIO \
    --score_dropout $SCORE_DROPOUT \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --eval_every $EVAL_EVERY \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_steps $NUM_STEPS \
    --decay_type $DECAY_TYPE \
    --warmup_steps $WARMUP_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --seed $SEED \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
"

# Optional flags
[ ! -z "$PRETRAINED_DIR" ] && COMMON_ARGS="$COMMON_ARGS --pretrained_dir $PRETRAINED_DIR"
[ ! -z "$VAL_ROOT" ]       && COMMON_ARGS="$COMMON_ARGS --val_root $VAL_ROOT"
[ ! -z "$LEARN_CURV" ]     && COMMON_ARGS="$COMMON_ARGS $LEARN_CURV"
[ ! -z "$PL_SAMPLE" ]      && COMMON_ARGS="$COMMON_ARGS $PL_SAMPLE"
[ ! -z "$FP16" ]            && COMMON_ARGS="$COMMON_ARGS $FP16"
[ ! -z "$USE_WANDB" ]      && COMMON_ARGS="$COMMON_ARGS $USE_WANDB --wandb_project $WANDB_PROJECT"
[ ! -z "$WANDB_ENTITY" ]   && COMMON_ARGS="$COMMON_ARGS --wandb_entity $WANDB_ENTITY"

if [ "$MODE" = "single" ]; then
    echo "=============================="
    echo "Hyperbolic PL-Only 학습 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "Loss: Plackett-Luce only (no entailment)"
    echo "Embed dim: $EMBED_DIM"
    echo "Score Head: layers=$SCORE_N_LAYERS, heads=$SCORE_N_HEADS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 train_hyperbolic_pl_only.py $COMMON_ARGS

elif [ "$MODE" = "multi" ]; then
    echo "=============================="
    echo "분산 학습 (Multi-GPU) 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "GPU 수: $NUM_GPUS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --nproc_per_node=$NUM_GPUS \
        train_hyperbolic_pl_only.py $COMMON_ARGS

else
    echo "MODE를 'single' 또는 'multi'로 설정해주세요."
    exit 1
fi