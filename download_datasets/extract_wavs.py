#!/usr/bin/env python
"""
Extract audio from HuggingFace AudioSet parquet files, resample to 16kHz,
save as individual wav files, and create a manifest JSON.

Usage:
    python extract_wavs.py \
        --dataset_name agkphysics/AudioSet \
        --dataset_config balanced \
        --cache_dir /ucappell/datasets/audioset \
        --output_dir /ucappell/datasets/audioset_wavs/balanced \
        --sample_rate 16000

Output structure:
    output_dir/
        wavs/
            --PJHxphWEs.wav
            --aE2O5G5WE.wav
            ...
        manifest.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Monkey-patch for datasets 3.6.0 compatibility if needed
try:
    from datasets import load_dataset
    # Quick test
    _ = load_dataset
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Extract wav files from HuggingFace AudioSet")
    parser.add_argument("--dataset_name", type=str, default="agkphysics/AudioSet")
    parser.add_argument("--dataset_config", type=str, default="balanced",
                        choices=["balanced", "unbalanced"])
    parser.add_argument("--cache_dir", type=str, default="/ucappell/datasets/audioset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for extracted wavs and manifest")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate for saved wav files")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        help="Which splits to extract")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for extraction")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config})")
    print(f"Cache dir: {args.cache_dir}")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
    )
    print(f"Available splits: {list(dataset.keys())}")

    for split in args.splits:
        if split not in dataset:
            print(f"  [SKIP] Split '{split}' not found in dataset.")
            continue

        split_data = dataset[split]
        split_output_dir = os.path.join(args.output_dir, split)
        wav_dir = os.path.join(split_output_dir, "wavs")
        os.makedirs(wav_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Extracting split: {split} ({len(split_data)} examples)")
        print(f"  Output: {split_output_dir}")
        print(f"  Target sample rate: {args.sample_rate} Hz")
        print(f"{'='*60}\n")

        manifest = []
        skipped = 0
        start_time = time.time()

        for i in tqdm(range(len(split_data)), desc=f"Extracting {split}"):
            example = split_data[i]

            # Extract audio
            audio = example["audio"]
            if isinstance(audio, dict):
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
            else:
                print(f"  [WARN] Unexpected audio format at index {i}, skipping.")
                skipped += 1
                continue

            # Convert to torch tensor
            waveform = torch.tensor(waveform, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [1, samples]
            elif waveform.ndim == 2 and waveform.shape[0] > 1:
                # Stereo to mono
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != args.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)

            # Generate filename from video_id (sanitize for filesystem)
            video_id = example.get("video_id", f"sample_{i:08d}")
            # Replace characters that might cause issues in filenames
            safe_id = video_id.replace("/", "_").replace("\\", "_")
            wav_filename = f"{safe_id}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            # Handle duplicate video_ids (some AudioSet clips share video_id)
            if os.path.exists(wav_path):
                wav_filename = f"{safe_id}_{i}.wav"
                wav_path = os.path.join(wav_dir, wav_filename)

            # Save as 16-bit wav
            torchaudio.save(wav_path, waveform, args.sample_rate, bits_per_sample=16)

            # Build manifest entry
            entry = {
                "wav": wav_path,
                "video_id": video_id,
                "duration": waveform.shape[-1] / args.sample_rate,
                "sample_rate": args.sample_rate,
            }

            # Preserve labels if present
            if "labels" in example:
                entry["labels"] = example["labels"]
            if "human_labels" in example:
                entry["human_labels"] = example["human_labels"]

            manifest.append(entry)

            # Progress update every 5000 examples
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(split_data) - i - 1) / rate
                print(f"  [{i+1}/{len(split_data)}] "
                      f"{rate:.1f} examples/s, "
                      f"~{remaining/60:.1f} min remaining")

        # Save manifest
        manifest_path = os.path.join(split_output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"data": manifest}, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n  Done! Extracted {len(manifest)} files, skipped {skipped}.")
        print(f"  Time: {elapsed/60:.1f} minutes ({len(manifest)/elapsed:.1f} examples/s)")
        print(f"  Manifest saved to: {manifest_path}")

        # Print size stats
        wav_sizes = [os.path.getsize(entry["wav"]) for entry in manifest[:100]]
        if wav_sizes:
            avg_size = np.mean(wav_sizes)
            total_estimated = avg_size * len(manifest)
            print(f"  Avg wav size: {avg_size/1024:.1f} KB")
            print(f"  Estimated total size: {total_estimated/1e9:.2f} GB")

    print("\nAll done!")


if __name__ == "__main__":
    main()
