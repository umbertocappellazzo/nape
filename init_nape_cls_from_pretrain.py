#!/usr/bin/env python
"""
Convert a pretrained NapeForPreTraining checkpoint into an
NapeForClassification model with a freshly initialized classification head.

Example usage on Audioset:
    python init_nape_cls_from_pretrain.py \
        --pretrained_dir outputs/nape-base-pretrain/checkpoint-8500 \
        --num_labels 527 \
        --save_dir outputs/nape-base-sft-init \
        --use_ema

This loads the pretrained backbone (or its EMA), creates a classification model
with 527 labels (AudioSet), zero-initializes the classifier head, and saves locally.
"""

import argparse
import os

import torch
import torch.nn as nn

from models.configuration_nape import NapeConfig
from models.modeling_nape import (
    NapeForPreTraining,
    NapeForClassification,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build NapeForClassification from NapeForPreTraining."
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        required=True,
        help="Path to pretrained checkpoint directory (contains config.json + model weights).",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Path to load NapeConfig from. If None, uses --pretrained_dir.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=527,
        help="Number of labels for classification head (527 for AudioSet).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Local directory to save the classification model.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, load backbone from pytorch_model_ema.bin instead of default weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_dir = args.config_dir if args.config_dir is not None else args.pretrained_dir

    # ---- Load config ----
    print(f"Loading config from {config_dir}...")
    config = NapeConfig.from_pretrained(config_dir)
    config.num_labels = args.num_labels
    # Disable causal mask for fine-tuning (bidirectional attention)
    config.is_causal = False

    # ---- Load backbone weights ----
    if args.use_ema:
        ema_path = os.path.join(args.pretrained_dir, "pytorch_model_ema.bin")
        if not os.path.exists(ema_path):
            raise FileNotFoundError(f"EMA checkpoint not found at {ema_path}")
        print(f"Loading EMA checkpoint from {ema_path}...")
        ckpt = torch.load(ema_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # Strip "nape." prefix if present (EMA saves full model state)
        backbone_sd = {}
        for k, v in ckpt.items():
            if k.startswith("nape."):
                backbone_sd[k[len("nape."):]] = v
            elif k.startswith("prediction_head.") or k.startswith("multi_step_heads.") or k.startswith("flow_matching_loss.") or k.startswith("deep_supervision_heads.") or k.startswith("dino_loss."):
                # Skip pretraining-only heads (prediction projector + multi-step heads + diffusion MLP + deep supervision + DINO projection)
                continue
            else:
                backbone_sd[k] = v
    else:
        print(f"Loading pretrained model from {args.pretrained_dir}...")
        pretrain_model = NapeForPreTraining.from_pretrained(
            args.pretrained_dir,
            torch_dtype=torch.float32,
        )
        backbone_sd = pretrain_model.nape.state_dict()

    # ---- Create classification model ----
    print(f"Initializing NapeForClassification (num_labels={args.num_labels})...")
    cls_model = NapeForClassification(config)

    # ---- Copy backbone weights ----
    print("Copying backbone weights...")
    missing, unexpected = cls_model.nape.load_state_dict(backbone_sd, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing examples: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected examples: {unexpected[:5]}")

    # ---- Zero-initialize classifier head ----
    if isinstance(cls_model.classifier, nn.Linear):
        nn.init.zeros_(cls_model.classifier.weight)
        if cls_model.classifier.bias is not None:
            nn.init.zeros_(cls_model.classifier.bias)
        print("Zero-initialized classifier head.")

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving to {args.save_dir}...")
    cls_model.save_pretrained(args.save_dir)
    config.save_pretrained(args.save_dir)

    # Verify
    total_params = sum(p.numel() for p in cls_model.parameters())
    trainable_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print("\nModel saved successfully:")
    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.1f}M")
    print(f"  Config: is_causal={config.is_causal}, num_labels={config.num_labels}")


if __name__ == "__main__":
    main()
