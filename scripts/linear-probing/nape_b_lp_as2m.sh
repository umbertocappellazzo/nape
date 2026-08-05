#!/bin/bash
# ========================
# Audio NAPE Base — Linear Probing on AudioSet (AS-20K balanced)
# ========================
set -e

export NCCL_P2P_DISABLE=1

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EXPERIMENT_NAME="nape-large-linearprobe-AS2M"
WANDB_PROJECT="Nape-Pretrain"

# ---- Model ----
# Path to init model from init_nape_cls_from_pretrain.py
MODEL_NAME="outputs/??"

DATASET_CONFIG="unbalanced"
# ---- Data ----
DATASET_ROOT="/path/to/dataset"
TRAIN_MANIFEST="/path/to/manifest/${DATASET_CONFIG}/train/manifest_AS2M_train.json"
EVAL_MANIFEST="/path/to/manifest/${DATASET_CONFIG}/test/manifest_AS_test.json"

# ---- Output ----
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"


# ---- Training hyperparams (following NAPE SFT recipe) ----
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=16
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))
NUM_EPOCHS=20
BASE_LEARNING_RATE=1e-2
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 64)")

# ========================
export WANDB_PROJECT=$WANDB_PROJECT
# ========================

echo "============================================"
echo "  Audio NAPE Base — Linear Probing"
echo "  GPUs: $NGPU"
echo "  Model: $MODEL_NAME"
echo "  Train manifest: $TRAIN_MANIFEST"
echo "  Eval manifest: $EVAL_MANIFEST"
echo "  Batch size: $TOTAL_BATCH_SIZE (per device: $PER_DEVICE_BATCH_SIZE, grad accum: $GRAD_ACCUM_STEPS)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output: $OUTPUT_DIR"
echo "============================================"


torchrun \
    --nproc_per_node $NGPU run_nape_sft.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --model_name_or_path $MODEL_NAME \
    --freeze_backbone True \
    --freeze_embed True \
    --probe_layer 11 \
    --drop_path_prob 0. \
    --use_ema True \
    --ema_decay 0.99995 \
    --bidirectional True \
    --pooling_mode mean \
    --label_smoothing 0.0 \
    \
    --train_manifest $TRAIN_MANIFEST \
    --eval_manifest $EVAL_MANIFEST \
    --dataset_root ${DATASET_ROOT} \
    --head_lr 1e-2 \
    --sampling_strategy inverse_frequency \
    --use_llrd False \
    --llrd 0.7 \
    \
    --freq_mask_param 0 \
    --time_mask_param 0 \
    --num_freq_masks 0 \
    --num_time_masks 0 \
    --mixup_alpha 0. \
    --mixup_prob 0. \
    --cutmix_alpha 0. \
    --cutmix_prob 0. \
    --use_audio_rolling False \
    --use_random_noise False \
    --noise_scale 0. \
    \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    --dataloader_drop_last True \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --lr_scheduler_kwargs '{"custom_scheduler_type": "llrd_cosine_warmup"}' \
    --warmup_ratio 0.2 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 20 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_ema_mAP \
    --greater_is_better True \
    --save_total_limit 1 \
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