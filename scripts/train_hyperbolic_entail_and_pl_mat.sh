#!/bin/bash

# ========================
# Multi-Scale Hyperbolic Entailment + Plackett-Luce (MAT v2) Training
# ========================
# Architecture:
#   ViT -> MERUStyleProjection -> [Pre-split LorentzBlocks] -> Multi-scale split
#   - Pre-split LorentzBlocks: Lorentzian attention at full dim D
#     (frame interaction ON the hyperboloid before splitting)
#   - Multi-scale: D, D/2, D/4 with per-scale LorentzScoreHead
#   - Scale weights: LEARNABLE via softmax over logits
# ========================
NAME="cholec80-hyperbolic-entail-pl-mat"
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
EMBED_DIM=128              # Full-scale dim (D). D/2=64, D/4=32. Must be divisible by 4.
CURV_INIT=1.0
LEARN_CURV="--learn_curv"

# Pre-split Lorentz interaction (Lorentzian attention at full D before splitting)
PRE_SPLIT_N_LAYERS=2       # Number of LorentzBlock layers before multi-scale split
PRE_SPLIT_N_HEADS=4        # Attention heads in pre-split blocks
PRE_SPLIT_MLP_RATIO=4.0
PRE_SPLIT_DROPOUT=0.1

# Entailment loss
MIN_RADIUS=0.1
HEIGHT_MARGIN=0.1
HEIGHT_WEIGHT=1.0
CONE_WEIGHT=1.0

# Plackett-Luce loss
PL_WEIGHT=1.0
PL_SAMPLE=""                # "--pl_sample" for subset sampling
PL_R=4
PL_K=8

# Learnable scale weights
# Scale weights are learned via softmax(logits / temperature).
# Temperature is also learnable. init_temp controls starting point.
SCALE_WEIGHT_TEMP=1.0      # Initial temperature (higher = more uniform start)

# Per-scale Lorentz Score Head
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
FP16=""                     # "--fp16" for mixed precision

# Wandb
USE_WANDB="--use_wandb"
WANDB_PROJECT="hyperbolic-temporal-vit"
# WANDB_ENTITY=""

# GPU
GPU_IDS="0"
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
    --pre_split_n_layers $PRE_SPLIT_N_LAYERS \
    --pre_split_n_heads $PRE_SPLIT_N_HEADS \
    --pre_split_mlp_ratio $PRE_SPLIT_MLP_RATIO \
    --pre_split_dropout $PRE_SPLIT_DROPOUT \
    --min_radius $MIN_RADIUS \
    --height_margin $HEIGHT_MARGIN \
    --height_weight $HEIGHT_WEIGHT \
    --cone_weight $CONE_WEIGHT \
    --pl_weight $PL_WEIGHT \
    --pl_R $PL_R \
    --pl_K $PL_K \
    --scale_weight_temp $SCALE_WEIGHT_TEMP \
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
    echo "Multi-Scale Hyperbolic + PL (MAT v2) training"
    echo "GPU: $GPU_IDS"
    echo "Loss: Entailment(cone=$CONE_WEIGHT, height=$HEIGHT_WEIGHT) + PL($PL_WEIGHT)"
    echo "Embed dim: $EMBED_DIM (scales: D, D/2, D/4)"
    echo "Pre-split LorentzBlocks: layers=$PRE_SPLIT_N_LAYERS, heads=$PRE_SPLIT_N_HEADS"
    echo "Scale weights: LEARNABLE (temp_init=$SCALE_WEIGHT_TEMP)"
    echo "Score Head: layers=$SCORE_N_LAYERS, heads=$SCORE_N_HEADS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 train_hyperbolic_entail_and_pl_mat.py $COMMON_ARGS

elif [ "$MODE" = "multi" ]; then
    echo "=============================="
    echo "Multi-GPU training"
    echo "GPU: $GPU_IDS, Num GPUs: $NUM_GPUS"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --nproc_per_node=$NUM_GPUS \
        train_hyperbolic_entail_and_pl_mat.py $COMMON_ARGS

else
    echo "Set MODE to 'single' or 'multi'."
    exit 1
fi