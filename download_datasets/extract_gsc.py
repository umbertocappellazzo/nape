#!/usr/bin/env python
"""
Extract audio from HuggingFace google/speech_commands parquet files, pad all
clips to exactly 1 second at 16 kHz, save as individual wav files, and create
per-split manifest JSONs plus a frozen classes.json.

Usage (v0.02 — 35 classes):
    python extract_gsc.py \
        --version v0.02 \
        --cache_dir /ucappell/datasets/gsc_hf \
        --output_dir /ucappell/datasets/gsc_v2

Usage (v0.01 — 30 classes):
    python extract_gsc.py \
        --version v0.01 \
        --cache_dir /ucappell/datasets/gsc_hf \
        --output_dir /ucappell/datasets/gsc_v1

Output structure:
    output_dir/
        wavs/                                 <- flat, shared across splits
            yes_abc12345_0.wav
            ...
        train/manifest.json
        validation/manifest.json
        test/manifest.json
        classes.json                          <- frozen 30 or 35 class list

Protocol: 35-class (v0.02) / 30-class (v0.01). The _silence_ class is excluded;
auxiliary ("unknown") words are kept as regular target classes. This matches
the convention used by AST, EAT, BEAT and most audio-SSL benchmarks.

Manifest schema per entry (identical format to AudioSet and ESC-50 manifests):
    {"wav": "<abs path>",
     "id": "yes_abc12345_0",
     "duration": 1.0,
     "sample_rate": 16000,
     "labels": ["yes"]}            # always a 1-elem list for single-label
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


SILENCE_LABEL = "_silence_"


def main():
    parser = argparse.ArgumentParser(description="Extract wav files from HuggingFace speech_commands")
    parser.add_argument("--dataset_name", type=str, default="google/speech_commands")
    parser.add_argument("--version", type=str, required=True, choices=["v0.01", "v0.02"],
                        help="Speech Commands version: v0.01 (30 classes) or v0.02 (35 classes)")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for extracted wavs and manifests")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "validation", "test"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Allow HF to run the speech_commands.py loader script. "
                             "Required because google/speech_commands is distributed as a "
                             "loader-script dataset (not static parquet). Default: True.")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset_name} ({args.version})")
    print(f"Cache dir: {args.cache_dir}")

    # Load all three splits up-front — we need the full label set across splits
    # to pin down the class list deterministically.
    dataset = load_dataset(
        args.dataset_name,
        args.version,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Available splits: {list(dataset.keys())}")

    # ---- Resolve the class list (integer → string via ClassLabel feature) ----
    # The HF feature gives us the canonical label ordering (10 core words,
    # then auxiliary words, then _silence_). We then strip _silence_ to get
    # the 30- or 35-class list.
    label_feature = dataset[args.splits[0]].features["label"]
    all_label_names = list(label_feature.names)
    print(f"Full HF label set ({len(all_label_names)}): {all_label_names}")

    if SILENCE_LABEL not in all_label_names:
        raise ValueError(
            f"Expected '{SILENCE_LABEL}' in label set. This extractor's filtering "
            f"logic assumes it exists. If HF changed the schema, review before running."
        )
    silence_idx = all_label_names.index(SILENCE_LABEL)
    kept_labels = [l for l in all_label_names if l != SILENCE_LABEL]

    expected = 30 if args.version == "v0.01" else 35
    if len(kept_labels) != expected:
        raise ValueError(
            f"Expected {expected} classes for {args.version} after stripping "
            f"{SILENCE_LABEL}, but got {len(kept_labels)}: {kept_labels}"
        )
    print(f"Kept classes ({len(kept_labels)}): {kept_labels}")

    # ---- Set up output dirs + save classes.json ----
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    classes_path = os.path.join(args.output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump({
            "classes": kept_labels,
            "num_classes": len(kept_labels),
            "version": args.version,
            "protocol": f"{len(kept_labels)}-class (silence excluded)",
        }, f, indent=2)
    print(f"Wrote {classes_path}")

    target_samples = args.sample_rate  # 1 second at 16 kHz = 16000 samples

    total_extracted = 0
    total_skipped_silence = 0

    for split in args.splits:
        if split not in dataset:
            print(f"  [SKIP] Split '{split}' not found.")
            continue

        split_data = dataset[split]
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Extracting split: {split} ({len(split_data)} examples)")
        print(f"  Output manifest: {split_dir}/manifest.json")
        print(f"  Target: {target_samples} samples ({target_samples/args.sample_rate:.2f}s) at {args.sample_rate} Hz")
        print(f"{'='*60}\n")

        manifest = []
        skipped_silence = 0
        skipped_other = 0
        start_time = time.time()

        for i in tqdm(range(len(split_data)), desc=f"Extracting {split}"):
            example = split_data[i]

            # Skip silence entries (we follow the 35-class / 30-class protocol)
            if example["label"] == silence_idx:
                skipped_silence += 1
                continue

            audio = example["audio"]
            if not isinstance(audio, dict):
                print(f"  [WARN] Unexpected audio format at index {i}, skipping.")
                skipped_other += 1
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

            # Pad or trim to exactly 1 second.
            # GSC real-command clips are nominally 1s but not all samples are exactly 16000.
            cur_len = waveform.shape[-1]
            if cur_len < target_samples:
                # Right-pad with zeros (matches the standard KWS preprocessing convention)
                pad_right = target_samples - cur_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_right))
            elif cur_len > target_samples:
                # Trim — shouldn't happen for real commands, but guard anyway
                waveform = waveform[:, :target_samples]

            # Filename: use label + speaker_id + utterance_id for uniqueness
            label_str = label_feature.int2str(example["label"])
            speaker_id = example.get("speaker_id", "unk")
            utterance_id = example.get("utterance_id", 0)
            # Some entries may have empty/None speaker_id
            if speaker_id is None or speaker_id == "":
                speaker_id = "unk"
            safe_id = f"{label_str}_{speaker_id}_{utterance_id}"
            # Guard against pathological characters
            safe_id = safe_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

            wav_filename = f"{safe_id}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            # Handle duplicates (same speaker+utterance can appear across splits
            # — shouldn't, but defensive coding is cheap here)
            if os.path.exists(wav_path):
                wav_filename = f"{safe_id}_{split}_{i}.wav"
                wav_path = os.path.join(wav_dir, wav_filename)

            torchaudio.save(wav_path, waveform, args.sample_rate, bits_per_sample=16)

            entry = {
                "wav": os.path.abspath(wav_path),
                "id": os.path.splitext(wav_filename)[0],
                "duration": float(target_samples / args.sample_rate),
                "sample_rate": args.sample_rate,
                "labels": [label_str],
            }
            manifest.append(entry)

        # Save per-split manifest
        manifest_path = os.path.join(split_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"data": manifest}, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n  {split}: extracted {len(manifest)}, "
              f"skipped {skipped_silence} silence + {skipped_other} other.")
        print(f"  Time: {elapsed/60:.1f} min "
              f"({len(manifest)/max(elapsed, 1e-6):.1f} examples/s)")
        print(f"  Manifest: {manifest_path}")

        total_extracted += len(manifest)
        total_skipped_silence += skipped_silence

    print(f"\n{'='*60}")
    print(f"  Summary for {args.version}:")
    print(f"  Total extracted: {total_extracted}")
    print(f"  Total silence skipped: {total_skipped_silence}")
    print(f"  Classes: {len(kept_labels)}")
    print(f"{'='*60}")

    # Size stats
    first_split = args.splits[0]
    first_manifest_path = os.path.join(args.output_dir, first_split, "manifest.json")
    if os.path.exists(first_manifest_path):
        with open(first_manifest_path) as f:
            sample_manifest = json.load(f)["data"]
        wav_sizes = [os.path.getsize(entry["wav"]) for entry in sample_manifest[:100]]
        if wav_sizes:
            avg_size = np.mean(wav_sizes)
            total_estimated = avg_size * total_extracted
            print(f"  Avg wav size: {avg_size/1024:.1f} KB")
            print(f"  Estimated total size: {total_estimated/1e9:.2f} GB")

    print("\nAll done!")


if __name__ == "__main__":
    main()
