set -e

export NCCL_P2P_DISABLE=1

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EXPERIMENT_NAME="nape-small-finetune-gscv1"
WANDB_PROJECT=""

# ---- Model ----
MODEL_NAME="outputs/??"

# ---- Data ----
DATASET_ROOT="/path/to/dataset"
TRAIN_MANIFEST="/path/to/manifest/train/manifest_gscv1_train.json"
EVAL_MANIFEST="/path/to/manifest/validation/manifest_gscv1_val.json"
TEST_MANIFEST="/path/to/manifest/test/manifest_gscv1_test.json"
CLASSES_FILE="/path/to/classes/classes_gscv1.json"
NORM_STATS_FILE="/path/to/stats/norm_stats_gscv1.json"

NORM_MEAN=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['mean'])")
NORM_STD=$(python -c "import json; print(json.load(open('${NORM_STATS_FILE}'))['std'])")

# ---- Output ----
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

# ---- Training hyperparams ----
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))
NUM_EPOCHS=50
BASE_LEARNING_RATE=5e-3
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")


export WANDB_PROJECT=$WANDB_PROJECT

echo "============================================"
echo "  Audio NAPE Small — GSC v1 KS1 (12-class)"
echo "  GPUs:          $NGPU"
echo "  Model:         $MODEL_NAME"
echo "  Train:         $TRAIN_MANIFEST"
echo "  Eval:          $EVAL_MANIFEST"
echo "  Classes:       $CLASSES_FILE"
echo "  Norm mean/std: $NORM_MEAN / $NORM_STD"
echo "  Sampling:      beats_style"
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
    --mixup_prob 0.8 \
    --cutmix_alpha 1.0 \
    --cutmix_prob 0.8 \
    --bidirectional True \
    --pooling_mode mean \
    --label_smoothing 0.1 \
    --use_audio_rolling False \
    --use_random_noise True \
    --noise_scale 0.1 \
    --num_register_tokens 0 \
    \
    --train_manifest $TRAIN_MANIFEST \
    --dataset_root ${DATASET_ROOT} \
    --eval_manifest $EVAL_MANIFEST \
    --test_manifest $TEST_MANIFEST \
    --classes_file $CLASSES_FILE \
    --task_type single_label \
    --audio_duration 1.0 \
    --sampling_strategy beats_style \
    --beats_unknown_class _unknown_ \
    --norm_mean $NORM_MEAN \
    --norm_std $NORM_STD \
    --head_lr 5e-3 \
    --use_llrd True \
    --llrd 0.6 \
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
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
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