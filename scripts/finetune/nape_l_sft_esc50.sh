set -e

export NCCL_P2P_DISABLE=1

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EXPERIMENT_BASE="nape-large-finetune-esc50"
WANDB_PROJECT=""

# ---- Model (cls-initialized, 50 labels) ----
MODEL_NAME="outputs/??"

# ---- Data ----
DATASET_ROOT="/path/to/dataset"
MANIFEST="/path/to/manifest/manifest_esc50.json"
CLASSES_FILE="/path/to/classes/classes_esc50.json"
NORM_STATS_FILE="/path/to/stats//norm_stats_esc50.json"

# Pull mean/std out of the norm_stats.json written by compute_norm_stats.py.
# These are specific to ESC-50's fbank distribution.
NORM_MEAN=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['mean'])")
NORM_STD=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['std'])")

# ---- Training hyperparams ----
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))
NUM_EPOCHS=100
BASE_LEARNING_RATE=5e-3
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")
DATALOADER_NUM_WORKERS=$((4 * NGPU))

export WANDB_PROJECT=$WANDB_PROJECT

echo "============================================"
echo "  Audio NAPE Large — ESC-50 5-fold CV"
echo "  GPUs:           $NGPU"
echo "  Model:          $MODEL_NAME"
echo "  Manifest:       $MANIFEST"
echo "  Classes file:   $CLASSES_FILE"
echo "  Norm mean/std:  $NORM_MEAN / $NORM_STD"
echo "  Batch size:     $TOTAL_BATCH_SIZE (per device: $PER_DEVICE_BATCH_SIZE, grad accum: $GRAD_ACCUM_STEPS)"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Learning rate:  $LEARNING_RATE"
echo "============================================"


# ---- Fold loop ----
# Each fold writes to outputs/${EXPERIMENT_BASE}/fold{N}/. A failed experiment
# can be cleaned up with a single `rm -rf outputs/${EXPERIMENT_BASE}/`.
# Each fold is an independent wandb run; review the best metrics per fold
# in wandb and compute the mean yourself.


for FOLD in 1 2 3 4 5; do
    RUN_NAME="${EXPERIMENT_BASE}-fold${FOLD}"
    OUTPUT_DIR="outputs/${EXPERIMENT_BASE}/fold${FOLD}"

    echo ""
    echo "============================================"
    echo "  ESC-50 fine-tuning — FOLD ${FOLD} of 5"
    echo "  Output:       $OUTPUT_DIR"
    echo "============================================"

    torchrun \
        --nproc_per_node $NGPU run_nape_sft.py \
        \
        --ddp_backend nccl \
        --ddp_find_unused_parameters False \
        \
        --model_name_or_path $MODEL_NAME \
        --freeze_backbone False \
        --freeze_embed False \
        --use_ema False \
        --ema_decay 0.99 \
        --drop_path_prob 0.1 \
        --mixup_alpha 0.8 \
        --mixup_prob 0.5 \
        --cutmix_alpha 1.0 \
        --cutmix_prob 0.5 \
        --bidirectional True \
        --pooling_mode mean \
        --label_smoothing 0.1 \
        --use_audio_rolling True \
        --use_random_noise True \
        --noise_scale 0.1 \
        --num_register_tokens 0 \
        \
        --train_manifest $MANIFEST \
        --dataset_root ${DATASET_ROOT} \
        --eval_manifest $MANIFEST \
        --classes_file $CLASSES_FILE \
        --task_type single_label \
        --audio_duration 5.0 \
        --fold_held_out ${FOLD} \
        --norm_mean $NORM_MEAN \
        --norm_std $NORM_STD \
        --head_lr 5e-3 \
        --use_llrd True \
        --llrd 0.9 \
        \
        --freq_mask_param 16 \
        --time_mask_param 24 \
        --num_freq_masks 2 \
        --num_time_masks 2 \
        \
        --do_train \
        --do_eval \
        --output_dir $OUTPUT_DIR \
        --remove_unused_columns False \
        --dataloader_drop_last True \
        --overwrite_output_dir True \
        \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE \
        --lr_scheduler_type cosine \
        --lr_scheduler_kwargs '{"custom_scheduler_type": "llrd_cosine_warmup"}' \
        --warmup_ratio 0.1 \
        --weight_decay 0.05 \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --optim adamw_torch \
        \
        --logging_strategy steps \
        --logging_steps 20 \
        --eval_strategy epoch \
        --save_strategy no \
        --load_best_model_at_end False \
        \
        --seed 555 \
        --bf16 True \
        \
        --dataloader_num_workers $DATALOADER_NUM_WORKERS \
        --dataloader_persistent_workers True \
        --dataloader_pin_memory False \
        \
        --report_to wandb \
        --run_name $RUN_NAME
done

echo ""
echo "============================================"
echo "  All 5 folds complete."
echo "  Per-fold results in outputs/${EXPERIMENT_BASE}/fold{1..5}/"
echo "  Per-fold wandb runs: ${EXPERIMENT_BASE}-fold{1..5}"
echo "============================================"