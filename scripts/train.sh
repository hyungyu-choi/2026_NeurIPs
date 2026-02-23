#!/bin/bash

# ========================
# Temporal Ordering ViT - 학습 설정
# ========================
NAME="cholec80-temporal-ordering"
MODEL_TYPE="ViT-B_16"
# PRETRAINED_DIR="checkpoint/ViT-B_16.npz"   # pretrained 사용 시 주석 해제
OUTPUT_DIR="output"

# Dataset
DATA_ROOT="../code/Dataset/cholec80/frames/extract_1fps/training_set"
# VAL_ROOT="../code/Dataset/cholec80/frames/extract_1fps/val_set"   # validation 있으면 주석 해제

# Temporal sampling
IMG_SIZE=224
SEQ_LEN=8
MIN_STEP=1
MAX_STEP=20
SAMPLING_MODE="global"    # "randstep" or "global"

# Temporal Head
HIDDEN_MUL=0.5
TEMPORAL_MAX_LEN=64
TEMPORAL_N_LAYERS=6
TEMPORAL_N_HEAD=8
TEMPORAL_DROPOUT=0.1

# Plackett-Luce loss
PL_SAMPLE=""                # "--pl_sample" 로 설정하면 subset sampling 사용
PL_R=4
PL_K=8

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

# ========================
# GPU 설정
# ========================
GPU_IDS="3"
MODE="single"               # "single" 또는 "multi"

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
    --hidden_mul $HIDDEN_MUL \
    --temporal_max_len $TEMPORAL_MAX_LEN \
    --temporal_n_layers $TEMPORAL_N_LAYERS \
    --temporal_n_head $TEMPORAL_N_HEAD \
    --temporal_dropout $TEMPORAL_DROPOUT \
    --pl_R $PL_R \
    --pl_K $PL_K \
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
[ ! -z "$PL_SAMPLE" ]      && COMMON_ARGS="$COMMON_ARGS $PL_SAMPLE"
[ ! -z "$FP16" ]            && COMMON_ARGS="$COMMON_ARGS $FP16"

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