#!/bin/bash

# ========================
# MERU-style Hyperbolic Entailment ViT - 학습 설정
# ========================
NAME="cholec80-hyperbolic-meru"
MODEL_TYPE="ViT-B_16"
# PRETRAINED_DIR="checkpoint/ViT-B_16.npz"   # pretrained 사용 시 주석 해제
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
EMBED_DIM=128              # 하이퍼볼릭 임베딩 차원 (space components only)
CURV_INIT=1.0              # 초기 곡률 (양수, hyperboloid curvature = -curv)
LEARN_CURV="--learn_curv"  # 곡률 학습 여부 (빈 문자열이면 고정)

# Entailment loss
MIN_RADIUS=0.1             # half-aperture의 min_radius (MERU의 K 파라미터)
HEIGHT_MARGIN=0.1          # height ordering margin
HEIGHT_WEIGHT=1.0          # height loss 가중치
CONE_WEIGHT=1.0            # cone loss 가중치

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
USE_WANDB="--use_wandb"    # 빈 문자열이면 wandb 비활성화
WANDB_PROJECT="hyperbolic-temporal-vit"
# WANDB_ENTITY=""           # wandb team/username (옵션)

# ========================
# GPU 설정
# ========================
GPU_IDS="1"
MODE="single"

# ========================
# 실행
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
    --min_radius $MIN_RADIUS \
    --height_margin $HEIGHT_MARGIN \
    --height_weight $HEIGHT_WEIGHT \
    --cone_weight $CONE_WEIGHT \
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
[ ! -z "$FP16" ]            && COMMON_ARGS="$COMMON_ARGS $FP16"
[ ! -z "$USE_WANDB" ]      && COMMON_ARGS="$COMMON_ARGS $USE_WANDB --wandb_project $WANDB_PROJECT"
[ ! -z "$WANDB_ENTITY" ]   && COMMON_ARGS="$COMMON_ARGS --wandb_entity $WANDB_ENTITY"

if [ "$MODE" = "single" ]; then
    echo "=============================="
    echo "MERU-style Hyperbolic Entailment 학습 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 train_hyperbolic.py $COMMON_ARGS

elif [ "$MODE" = "multi" ]; then
    echo "=============================="
    echo "분산 학습 (Multi-GPU) 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "GPU 수: $NUM_GPUS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --nproc_per_node=$NUM_GPUS \
        train_hyperbolic.py $COMMON_ARGS

else
    echo "MODE를 'single' 또는 'multi'로 설정해주세요."
    exit 1
fi