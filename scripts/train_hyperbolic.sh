#!/bin/bash

# ========================
# Hyperbolic Entailment ViT - 학습 설정
# ========================
NAME="cholec80-hyperbolic-entailment"
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

# Hyperbolic space
HYP_DIM=128               # 하이퍼볼릭 임베딩 차원 (출력은 HYP_DIM+1)
CURVATURE=1.0              # Lorentz model curvature K

# Entailment loss
HEIGHT_MARGIN=0.1          # height ordering margin
CONE_MARGIN=0.05           # cone inclusion margin
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

# ========================
# GPU 설정
# ========================
GPU_IDS="2"
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
    --hyp_dim $HYP_DIM \
    --curvature $CURVATURE \
    --height_margin $HEIGHT_MARGIN \
    --cone_margin $CONE_MARGIN \
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
[ ! -z "$FP16" ]            && COMMON_ARGS="$COMMON_ARGS $FP16"

if [ "$MODE" = "single" ]; then
    echo "=============================="
    echo "Hyperbolic Entailment 학습 시작"
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