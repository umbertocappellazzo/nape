set -e

export NCCL_P2P_DISABLE=1

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EXPERIMENT_BASE="nape-large-finetune-iemocap"
WANDB_PROJECT=""

# ---- Model ----
MODEL_NAME="outputs/??"

# ---- Data ----
DATASET_ROOT="/path/to/dataset"
TRAIN_MANIFEST="/path/to/manifest/manifest_iemocap.json"
EVAL_MANIFEST="/path/to/manifest/manifest_iemocap.json"
CLASSES_FILE="/path/to/classes/classes_iemocap.json"
NORM_STATS_FILE="/path/to/stats/norm_stats_iemocap.json"

NORM_MEAN=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['mean'])")
NORM_STD=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['std'])")

# ---- Training hyperparams ----
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=16
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))
NUM_EPOCHS=50
BASE_LEARNING_RATE=5e-3
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")
DATALOADER_NUM_WORKERS=$((4 * NGPU))

export WANDB_PROJECT=$WANDB_PROJECT

# ---- 5-fold cross-validation loop ----
for FOLD in 1 2 3 4 5; do

    OUTPUT_DIR="outputs/${EXPERIMENT_BASE}/fold${FOLD}"
    RUN_NAME="${EXPERIMENT_BASE}-fold${FOLD}"

    echo "============================================"
    echo "  Audio NAPE Large — IEMOCAP 4-class"
    echo "  Fold:          ${FOLD} (held-out test session)"
    echo "  GPUs:          $NGPU"
    echo "  Model:         $MODEL_NAME"
    echo "  Manifest:      $TRAIN_MANIFEST"
    echo "  Classes:       $CLASSES_FILE"
    echo "  Norm mean/std: $NORM_MEAN / $NORM_STD"
    echo "  Sampling:      none (natural class frequencies)"
    echo "  Batch size:    $TOTAL_BATCH_SIZE (per device: $PER_DEVICE_BATCH_SIZE)"
    echo "  Epochs:        $NUM_EPOCHS"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Output:        $OUTPUT_DIR"
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
        --train_manifest $TRAIN_MANIFEST \
        --dataset_root ${DATASET_ROOT} \
        --eval_manifest $EVAL_MANIFEST \
        --classes_file $CLASSES_FILE \
        --fold_held_out $FOLD \
        --task_type single_label \
        --audio_duration 15.0 \
        --sampling_strategy none \
        --norm_mean $NORM_MEAN \
        --norm_std $NORM_STD \
        --head_lr 5e-3 \
        --use_llrd True \
        --llrd 0.9 \
        \
        --freq_mask_param 24 \
        --time_mask_param 48 \
        --num_freq_masks 2 \
        --num_time_masks 2 \
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

    echo ""
    echo "============================================"
    echo "  Finished fold ${FOLD}"
    echo "============================================"
    echo ""

done

echo ""
echo "All 5 folds complete. Per-fold final eval accuracies are visible in wandb"
echo "under runs ${EXPERIMENT_BASE}-fold1 ... -fold5. The reported 5-fold mean"
echo "is the average of the final-epoch eval_accuracy across all 5 runs."
