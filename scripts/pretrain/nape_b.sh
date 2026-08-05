#!/bin/bash
# ========================
# Audio NAPE — Base model pretraining
# ========================
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1


: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EXPERIMENT_NAME="nape-base-pretrain"
CONFIG_NAME="configs/nape-base-patch16"
DATASET_CONFIG="unbalanced"
TRAIN_MANIFEST="/path/to/manifest/${DATASET_CONFIG}/train/manifest_AS2M_train.json"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
WANDB_PROJECT=""
DATASET_ROOT="/path/to/dataset"

TOTAL_BATCH_SIZE=256
PER_DEVICE_BATCH_SIZE=32
GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE) ))
NUM_EPOCHS=30
BASE_LEARNING_RATE=5e-3
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / ${TOTAL_BATCH_SIZE})")

# ========================
export WANDB_PROJECT=$WANDB_PROJECT
# ========================
torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $NGPU run_audio_nape.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --config_name $CONFIG_NAME \
    --train_manifest $TRAIN_MANIFEST \
    --dataset_root ${DATASET_ROOT} \
    --dataloader_drop_last True \
    \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy epoch \
    --save_steps 1 \
    \
    --seed 555 \
    --bf16 True \
    \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \
    \
    --report_to wandb \
    --run_name $EXPERIMENT_NAME