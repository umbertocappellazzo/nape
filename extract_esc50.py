#!/usr/bin/env python
"""
Extract audio from HuggingFace ashraq/esc50 parquet files, resample to 16 kHz,
save as individual wav files, and create a master manifest JSON plus a frozen
classes.json mapping.

Usage:
    python extract_esc50.py \
        --cache_dir /ucappell/datasets/esc50_hf \
        --output_dir /ucappell/datasets/esc50 \
        --sample_rate 16000

Output structure:
    output_dir/
        wavs/
            1-100032-A-0.wav
            ...
        manifest.json        <- all 2000 entries, with "fold" field (1..5)
        classes.json         <- frozen 50-class list (alphabetical order)

The manifest schema is:
    {"data": [
        {"wav": "<abs path>",
         "id": "1-100032-A-0",
         "duration": 5.0,
         "sample_rate": 16000,
         "labels": ["dog"],          # always a list, 1-elem for single-label
         "fold": 1                   # 1..5, ESC-50 cross-val folds
        },
        ...
    ]}

The 50-class list is sorted alphabetically by category name to be deterministic
and stable across machines (the HF dataset's "target" integer uses a different
order — we do not trust it across reinstalls).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract wav files from HuggingFace ESC-50")
    parser.add_argument("--dataset_name", type=str, default="ashraq/esc50")
    parser.add_argument("--cache_dir", type=str, default="/ucappell/datasets/esc50_hf")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for extracted wavs and manifest")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate for saved wav files")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset_name}")
    print(f"Cache dir: {args.cache_dir}")
    dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        split="train",  # ESC-50 on HF is a single split named 'train'; folds are a column
    )
    print(f"Loaded {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")

    # ---- Build the frozen class list (alphabetical, deterministic) ----
    all_categories = sorted(set(dataset["category"]))
    assert len(all_categories) == 50, f"Expected 50 ESC-50 classes, got {len(all_categories)}"
    print(f"Classes ({len(all_categories)}): {all_categories[:5]} ... {all_categories[-3:]}")

    # ---- Sanity-check folds ----
    folds = sorted(set(int(f) for f in dataset["fold"]))
    assert folds == [1, 2, 3, 4, 5], f"Expected folds [1..5], got {folds}"

    # ---- Set up output dirs ----
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # ---- Save classes.json first (so a failed extraction still leaves behind the class list) ----
    classes_path = os.path.join(args.output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump({"classes": all_categories, "num_classes": len(all_categories)}, f, indent=2)
    print(f"Wrote {classes_path}")

    # ---- Extract ----
    print(f"\n{'='*60}")
    print(f"  Extracting ESC-50 ({len(dataset)} examples)")
    print(f"  Output: {args.output_dir}")
    print(f"  Target sample rate: {args.sample_rate} Hz")
    print(f"{'='*60}\n")

    manifest = []
    skipped = 0
    start_time = time.time()

    for i in tqdm(range(len(dataset)), desc="Extracting ESC-50"):
        example = dataset[i]

        audio = example["audio"]
        if not isinstance(audio, dict):
            print(f"  [WARN] Unexpected audio format at index {i}, skipping.")
            skipped += 1
            continue

        waveform = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != args.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)

        # Filename: use the original .wav basename (already unique in ESC-50)
        src_filename = example.get("filename", f"sample_{i:05d}.wav")
        safe_id = os.path.splitext(src_filename)[0].replace("/", "_").replace("\\", "_")
        wav_filename = f"{safe_id}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        # ESC-50 filenames are unique, so no duplicate handling needed — but guard anyway
        if os.path.exists(wav_path):
            wav_filename = f"{safe_id}_{i}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)

        torchaudio.save(wav_path, waveform, args.sample_rate, bits_per_sample=16)

        category = example["category"]
        assert category in all_categories, f"Unknown category: {category}"

        entry = {
            "wav": os.path.abspath(wav_path),
            "id": safe_id,
            "duration": float(waveform.shape[-1] / args.sample_rate),
            "sample_rate": args.sample_rate,
            "labels": [category],
            "fold": int(example["fold"]),
        }
        manifest.append(entry)

    # ---- Save manifest ----
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"data": manifest}, f, indent=2)

    elapsed = time.time() - start_time

    # ---- Summary ----
    print(f"\n  Done! Extracted {len(manifest)} files, skipped {skipped}.")
    print(f"  Time: {elapsed/60:.1f} minutes "
          f"({len(manifest)/max(elapsed, 1e-6):.1f} examples/s)")
    print(f"  Manifest saved to: {manifest_path}")
    print(f"  Classes saved to:  {classes_path}")

    # Per-fold counts
    print("\n  Per-fold counts (expected: ~400 each):")
    for fold in [1, 2, 3, 4, 5]:
        n = sum(1 for e in manifest if e["fold"] == fold)
        print(f"    fold {fold}: {n}")

    # Size stats
    wav_sizes = [os.path.getsize(entry["wav"]) for entry in manifest[:100]]
    if wav_sizes:
        avg_size = np.mean(wav_sizes)
        total_estimated = avg_size * len(manifest)
        print(f"\n  Avg wav size: {avg_size/1024:.1f} KB")
        print(f"  Estimated total size: {total_estimated/1e6:.1f} MB")

    print("\nAll done!")


if __name__ == "__main__":
    main()
