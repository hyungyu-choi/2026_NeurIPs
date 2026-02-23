#!/bin/bash

# ========================
# 학습 설정
# ========================
NAME="cifar10-run"
DATASET="cifar10"          # cifar10 or cifar100
MODEL_TYPE="ViT-B_16"
# PRETRAINED_DIR="checkpoint/ViT-B_16.npz"
OUTPUT_DIR="output"

IMG_SIZE=224
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=64
EVAL_EVERY=100

LEARNING_RATE=3e-2
WEIGHT_DECAY=0
NUM_STEPS=10000
DECAY_TYPE="cosine"
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0
SEED=42
GRADIENT_ACCUMULATION_STEPS=1

# ========================
# GPU 설정
# ========================
# 사용할 GPU 번호 (예: "0" / "0,1" / "0,1,2,3")
GPU_IDS="3"

# 학습 모드 선택: "single" 또는 "multi"
MODE="single"

# ========================
# 실행
# ========================
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

COMMON_ARGS="
    --name $NAME \
    --dataset $DATASET \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --img_size $IMG_SIZE \
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

# pretrained_dir가 설정된 경우에만 추가
if [ ! -z "$PRETRAINED_DIR" ]; then
    COMMON_ARGS="$COMMON_ARGS --pretrained_dir $PRETRAINED_DIR"
fi

if [ "$MODE" = "single" ]; then
    echo "=============================="
    echo "단일 GPU 학습 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 train.py $COMMON_ARGS

elif [ "$MODE" = "multi" ]; then
    echo "=============================="
    echo "분산 학습 (Multi-GPU) 시작"
    echo "사용 GPU: $GPU_IDS"
    echo "GPU 수: $NUM_GPUS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --nproc_per_node=$NUM_GPUS \
        train.py $COMMON_ARGS

else
    echo "MODE를 'single' 또는 'multi'로 설정해주세요."
    exit 1
fi