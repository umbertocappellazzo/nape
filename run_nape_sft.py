# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Nape on AudioSet classification (multi-label, mAP)."""

import json
import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from sklearn.metrics import average_precision_score

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names

from models import NapeForClassification, NapeConfig

warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrogram extraction (same as pretraining)
# ---------------------------------------------------------------------------

class SpectrogramExtractor:
    """Compute log-mel fbank features using Kaldi-compatible extraction."""

    def __init__(self, config: NapeConfig, norm_mean: float = -4.2677393, norm_std: float = 4.5689974):
        self.sample_rate = config.sample_rate
        self.target_time_frames = config.target_time_frames
        self.freq_bins = config.freq_bins
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.target_samples = int(config.audio_duration * config.sample_rate)

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            waveform = F.pad(waveform, (0, self.target_samples - num_samples))
        elif num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=self.sample_rate,
            frame_length=25.0,
            frame_shift=10.0,
            num_mel_bins=self.freq_bins,
            htk_compat=False,
            use_energy=False,
            window_type="hanning",
            dither=0.0,
        )

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        fbank = fbank.T.unsqueeze(0)

        time_frames = fbank.shape[-1]
        if time_frames < self.target_time_frames:
            fbank = F.pad(fbank, (0, self.target_time_frames - time_frames))
        elif time_frames > self.target_time_frames:
            fbank = fbank[:, :, :self.target_time_frames]

        return fbank


# ---------------------------------------------------------------------------
# Audio Augmentations for fine-tuning
# ---------------------------------------------------------------------------

class SpecAugment:
    """SpecAugment: time and frequency masking on spectrograms (matching AST/EAT)."""

    def __init__(
        self,
        freq_mask_param: int = 48,
        time_mask_param: int = 192,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: [1, freq_bins, time_frames]
        Returns:
            augmented spectrogram: [1, freq_bins, time_frames]
        """
        spec = spectrogram.clone()
        _, freq_bins, time_frames = spec.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f = min(f, freq_bins)
            if f > 0:
                f0 = torch.randint(0, freq_bins - f + 1, (1,)).item()
                spec[:, f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t = min(t, time_frames)
            if t > 0:
                t0 = torch.randint(0, time_frames - t + 1, (1,)).item()
                spec[:, :, t0:t0 + t] = 0.0

        return spec


class SpectrogramMixup:
    """Mixup on spectrograms with multi-label support (BCE-compatible)."""

    def __init__(self, alpha: float = 0.5, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, spectrograms: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            spectrograms: [B, 1, F, T]
            labels: [B, num_labels] multi-hot
        Returns:
            mixed spectrograms, mixed labels
        """
        if torch.rand(1).item() > self.prob:
            return spectrograms, labels

        batch_size = spectrograms.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)  # ensure lambda >= 0.5

        indices = torch.randperm(batch_size)
        spectrograms = lam * spectrograms + (1.0 - lam) * spectrograms[indices]
        labels = lam * labels + (1.0 - lam) * labels[indices]

        return spectrograms, labels


class SpectrogramCutMix:
    """
    CutMix on spectrograms with multi-label support (BCE-compatible).
    Cuts a random rectangular region from one spectrogram and pastes it onto another.
    Labels are mixed proportionally to the area ratio.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, spectrograms: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            spectrograms: [B, 1, F, T]
            labels: [B, num_labels] multi-hot
        Returns:
            cutmixed spectrograms, mixed labels
        """
        if torch.rand(1).item() > self.prob:
            return spectrograms, labels

        batch_size, _, freq_bins, time_frames = spectrograms.shape
        lam = np.random.beta(self.alpha, self.alpha)

        # Compute cut region size from lambda
        cut_ratio = np.sqrt(1.0 - lam)
        cut_f = int(freq_bins * cut_ratio)
        cut_t = int(time_frames * cut_ratio)

        # Random position for the cut
        f0 = np.random.randint(0, freq_bins - cut_f + 1) if cut_f < freq_bins else 0
        t0 = np.random.randint(0, time_frames - cut_t + 1) if cut_t < time_frames else 0

        indices = torch.randperm(batch_size)

        # Paste the cut region
        spectrograms = spectrograms.clone()
        spectrograms[:, :, f0:f0 + cut_f, t0:t0 + cut_t] = \
            spectrograms[indices, :, f0:f0 + cut_f, t0:t0 + cut_t]

        # Adjust lambda to actual area ratio
        lam_actual = 1.0 - (cut_f * cut_t) / (freq_bins * time_frames)
        labels = lam_actual * labels + (1.0 - lam_actual) * labels[indices]

        return spectrograms, labels


class AudioRolling:
    """
    Random circular shift along the time axis of the spectrogram.
    Follows AST/AudioMAE/EAT: torch.roll with random shift along time dimension.
    Applied per-sample in the dataset __getitem__.
    """

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: [1, freq_bins, time_frames]
        Returns:
            rolled spectrogram: [1, freq_bins, time_frames]
        """
        time_frames = spectrogram.shape[-1]
        shift = np.random.randint(-time_frames, time_frames)
        return torch.roll(spectrogram, shifts=shift, dims=-1)


class RandomNoise:
    """
    Additive random noise on the spectrogram.
    Follows AST/AudioMAE/EAT: fbank += torch.rand(...) * np.random.rand() / 10
    Applied per-sample in the dataset __getitem__.
    """

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: [1, freq_bins, time_frames]
        Returns:
            noisy spectrogram: [1, freq_bins, time_frames]
        """
        noise = torch.rand_like(spectrogram) * np.random.rand() * self.noise_scale
        return spectrogram + noise


# ---------------------------------------------------------------------------
# Wav Manifest Dataset for fine-tuning (with labels)
# ---------------------------------------------------------------------------

class WavManifestClassificationDataset(Dataset):
    """
    Loads wav files from manifest, computes spectrograms, and returns multi-hot labels.
    Supports:
      - ESC-50 style fold-based cross-validation via `fold_held_out` + `mode`.
      - GSC KS1 long-silence entries via `is_long_silence` flag (random 1s slice
        at load time).
      - An externally-provided `label2id` so the label space is stable regardless
        of which rows the fold filter keeps (important for cross-validation).
    """

    def __init__(
        self,
        manifest_path: str,
        spec_extractor: SpectrogramExtractor,
        label2id: dict,
        spec_augment: Optional[SpecAugment] = None,
        audio_rolling: Optional[AudioRolling] = None,
        random_noise: Optional[RandomNoise] = None,
        max_samples: Optional[int] = None,
        fold_held_out: int = 0,
        mode: str = "train",
        dataset_root: str = None,
    ):
        """
        Args:
            fold_held_out: If > 0, activate fold-based filtering. For ESC-50 style
                5-fold CV, set this to 1..5; `mode="train"` keeps entries with
                fold != fold_held_out, `mode="eval"` keeps entries with
                fold == fold_held_out. If 0, no filtering is applied (AudioSet / GSC).
            mode: "train" or "eval" — only consulted when fold_held_out > 0.
        """
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        all_entries = manifest["data"]

        # Fold filtering (ESC-50 style). When active, `fold_held_out` is
        # 1-indexed to match ESC-50 filename convention.
        if fold_held_out and fold_held_out > 0:
            if mode == "train":
                filtered = [e for e in all_entries if int(e.get("fold", -1)) != fold_held_out]
            elif mode == "eval":
                filtered = [e for e in all_entries if int(e.get("fold", -1)) == fold_held_out]
            else:
                raise ValueError(f"mode must be 'train' or 'eval' when fold_held_out > 0, got {mode}")
            logger.info(
                f"Fold filter active: fold_held_out={fold_held_out}, mode='{mode}', "
                f"{len(filtered)}/{len(all_entries)} entries retained."
            )
            self.entries = filtered
        else:
            self.entries = all_entries

        if max_samples is not None:
            self.entries = self.entries[:max_samples]

        self.spec_extractor = spec_extractor
        self.spec_augment = spec_augment
        self.audio_rolling = audio_rolling
        self.random_noise = random_noise
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.dataset_root = dataset_root

        # Precompute long-silence flag per entry (bool tensor for fast indexing).
        # In GSC KS1, long background-noise files are flagged to be randomly
        # sliced at load time.
        self._is_long_silence = [bool(e.get("is_long_silence", False)) for e in self.entries]
        n_long = sum(self._is_long_silence)
        if n_long > 0:
            logger.info(
                f"Long-silence entries in this split: {n_long} "
                f"(will be randomly sliced to {1.0}s at load time)."
            )

        logger.info(f"Loaded classification manifest: {manifest_path} "
                    f"({len(self.entries)} samples, {self.num_labels} labels)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav_path = os.path.join(self.dataset_root, entry["wav"])
        #print(f"[dataset] loading: {wav_path!r} (exists: {os.path.exists(wav_path)})")

        # Load wav
        waveform, sr = torchaudio.load(wav_path)

        # Long-silence random slicing (GSC KS1). We slice at the waveform level
        # BEFORE the spec extractor's pad/trim path, so a 60s background file
        # contributes a different 1s window on every draw. Uses torch.randint
        # so the RNG respects HF Trainer's per-worker seed for reproducibility.
        if self._is_long_silence[idx]:
            target_samples = int(self.spec_extractor.target_samples)
            cur_samples = waveform.shape[-1]
            if cur_samples > target_samples:
                max_start = cur_samples - target_samples
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start : start + target_samples]
            # If somehow shorter than target, fall through — the spec extractor
            # will zero-pad to target_samples.

        # Compute spectrogram
        spectrogram = self.spec_extractor(waveform, sr)

        # Audio rolling (circular time shift) — applied before masking (AST/EAT order)
        if self.audio_rolling is not None:
            spectrogram = self.audio_rolling(spectrogram)

        # Random noise — applied before masking (AST/EAT order)
        if self.random_noise is not None:
            spectrogram = self.random_noise(spectrogram)

        # Apply SpecAugment if in training mode
        if self.spec_augment is not None:
            spectrogram = self.spec_augment(spectrogram)

        # Multi-hot label encoding (works for both multi-label AudioSet and
        # single-label ESC-50/GSC — in the single-label case exactly one entry
        # will be 1.0, which mixup/cutmix soften into a probability vector that
        # CE-with-soft-targets handles natively).
        label_vec = torch.zeros(self.num_labels, dtype=torch.float32)
        if "labels" in entry:
            for label in entry["labels"]:
                if label in self.label2id:
                    label_vec[self.label2id[label]] = 1.0

        return {"spectrogram": spectrogram, "labels": label_vec}


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_audioset_label_map(manifest_path: str):
    """
    Build label2id and id2label mappings from manifest labels.
    Collects all unique labels across the manifest.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_labels = set()
    for entry in manifest["data"]:
        if "labels" in entry:
            for label in entry["labels"]:
                all_labels.add(label)

    all_labels = sorted(all_labels)
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


def load_label_map_from_classes_file(classes_path: str):
    """
    Load label2id / id2label from a `classes.json` file produced by the
    extract_esc50.py / extract_gsc.py / make_gsc_v1_ks1.py scripts.

    Expected format:
        {"classes": ["airplane", "breathing", ...], "num_classes": 50, ...}

    The class ORDER in the JSON is authoritative — this is what makes
    cross-fold label ids stable on ESC-50 (where a single fold might be
    missing some class entirely) and what gives GSC KS1 its canonical
    SUPERB ordering.
    """
    with open(classes_path, "r") as f:
        blob = json.load(f)
    classes = blob["classes"]
    assert isinstance(classes, list) and len(classes) > 0, \
        f"classes.json at {classes_path} must have a non-empty 'classes' list."
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"Loaded label map from {classes_path}: {len(classes)} classes.")
    return label2id, id2label


def _load_and_filter_entries(manifest_path: str, fold_held_out: int = 0, mode: str = "train"):
    """Shared helper: load a manifest and apply the same fold filter the dataset uses."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    entries = manifest["data"]
    if fold_held_out and fold_held_out > 0:
        if mode == "train":
            entries = [e for e in entries if int(e.get("fold", -1)) != fold_held_out]
        elif mode == "eval":
            entries = [e for e in entries if int(e.get("fold", -1)) == fold_held_out]
    return entries


def compute_sample_weights(
    manifest_path: str,
    label2id: dict,
    sampling_strategy: str = "inverse_frequency",
    beats_unknown_class: str = "_unknown_",
    fold_held_out: int = 0,
) -> np.ndarray:
    """
    Compute per-sample sampling weights for a `WeightedRandomSampler`.

    Supported strategies (always applied to the TRAIN split):

      "inverse_frequency"
          Original AST/PSLA/EAT scheme: weight = sum of (1 / class_count) over
          the labels of each sample. Appropriate for AudioSet multi-label
          with heavy class imbalance.

      "uniform_balanced"
          Every class contributes equal total mass per epoch. For single-label
          datasets this means each class gets sampled 1/K of the time,
          regardless of natural frequency. Effective when classes have
          roughly similar difficulty (ESC-50, balanced GSC-v2).

      "beats_style"
          BEATs convention for GSC KS1: every non-unknown class is allocated
          50% of the sampling mass that the unknown class gets. The unknown
          class keeps its natural (high) frequency; keyword and silence classes
          each get resampled to ~0.5 × count(unknown) presentations per epoch.
          With 10 keywords + 1 silence + 1 unknown, this makes silence and
          each keyword ~50% as prevalent as unknown, so unknown still
          dominates slightly but keywords are massively upweighted from their
          raw ~6% frequency.

      "none" / any other value → equal weights (effectively disabled).

    Long-silence handling:
        Entries with `is_long_silence=True` represent a POOL of random slices,
        not a fixed training example. Their weights are inflated so that the
        total mass assigned to silence equals the class's target mass —
        regardless of how few pool files exist. With 5 train silence files
        and a target mass of (say) 16500 draws per epoch, each long-silence
        file gets weight 16500 / 5 = 3300; each draw yields a different
        random 1s window.

    Fold filtering:
        When `fold_held_out > 0`, only the training-mode subset of the manifest
        is used for weight computation, so weights match the dataset that will
        be sampled from.

    Returns: np.ndarray of per-sample weights. Normalized so the mean is 1.0.
    """
    entries = _load_and_filter_entries(manifest_path, fold_held_out, mode="train")
    num_labels = len(label2id)
    n = len(entries)

    # Pass 1 — count, per class, how many entries carry that label. For single-
    # label datasets this is just a class histogram; for multi-label AudioSet
    # each entry contributes to multiple class counts.
    label_counts = np.zeros(num_labels, dtype=np.float64)
    for entry in entries:
        for label in entry.get("labels", []):
            if label in label2id:
                label_counts[label2id[label]] += 1.0
    label_counts = np.maximum(label_counts, 1.0)  # avoid div by zero

    # Per-class TARGET MASS per epoch. This is what determines how many draws
    # each class will receive on expectation. The specific value doesn't matter
    # after normalization, only the relative ratios across classes do.
    if sampling_strategy in ("none",):
        # Uniform per sample → every entry has weight 1.0 after normalization.
        return np.ones(n, dtype=np.float64)

    target_mass = np.zeros(num_labels, dtype=np.float64)

    if sampling_strategy == "inverse_frequency":
        # Every class contributes equal target mass. Same as uniform_balanced
        # on single-label; on multi-label AudioSet the aggregation in pass 2
        # gives entries with rare labels additive boosts.
        target_mass[:] = 1.0

    elif sampling_strategy == "uniform_balanced":
        # Same as inverse_frequency mathematically for single-label datasets —
        # kept as a separate name for clarity in shell scripts.
        target_mass[:] = 1.0

    elif sampling_strategy == "beats_style":
        # Unknown gets full natural mass (1.0), every other class gets 50%.
        unknown_id = label2id.get(beats_unknown_class, None)
        if unknown_id is None:
            raise ValueError(
                f"sampling_strategy='beats_style' requires a class named "
                f"'{beats_unknown_class}' in label2id; not found. "
                f"Available: {list(label2id.keys())[:10]}..."
            )
        # Expressed as per-class mass ratios; normalization later.
        target_mass[:] = 0.5
        target_mass[unknown_id] = 1.0

    else:
        raise ValueError(
            f"Unknown sampling_strategy: '{sampling_strategy}'. "
            f"Must be one of: none, inverse_frequency, uniform_balanced, beats_style."
        )

    # Per-entry weight: for each label the entry carries, add the class's
    # target_mass / count(class). For single-label data this is simply
    # (target_mass of that one class) / (its count).
    sample_weights = np.zeros(n, dtype=np.float64)
    for i, entry in enumerate(entries):
        for label in entry.get("labels", []):
            if label in label2id:
                cid = label2id[label]
                sample_weights[i] += target_mass[cid] / label_counts[cid]

    # Long-silence inflation: silence-pool files should be re-used as if the
    # pool were the full target count. `_is_long_silence=True` entries get
    # their weight inflated so the total mass on the silence class matches
    # what the strategy prescribes, not what 5 files would otherwise get.
    # Detection: any entry tagged `is_long_silence=True`. We treat these as
    # sharing the class mass of whatever class they carry (usually `_silence_`).
    long_silence_mask = np.array(
        [bool(e.get("is_long_silence", False)) for e in entries], dtype=bool
    )
    if long_silence_mask.any():
        # Group long-silence entries by class, then for each such class
        # redistribute its existing total mass equally among the long-silence
        # entries of that class.
        per_entry_class = np.full(n, -1, dtype=np.int64)
        for i, entry in enumerate(entries):
            for label in entry.get("labels", []):
                if label in label2id:
                    per_entry_class[i] = label2id[label]
                    break

        for cid in np.unique(per_entry_class[long_silence_mask]):
            if cid < 0:
                continue
            in_class = (per_entry_class == cid)
            long_in_class = in_class & long_silence_mask
            n_long = int(long_in_class.sum())
            if n_long == 0:
                continue
            # This class's total mass (across all its entries, long or short).
            total_mass = float(sample_weights[in_class].sum())
            # Distribute the class's full target mass to the long-silence
            # entries, giving effectively-infinite-pool behavior.
            # (Short entries of the same class keep their current weight.)
            if total_mass > 0:
                # Each long entry's weight becomes total_mass / n_long.
                sample_weights[long_in_class] = total_mass / n_long

    # Normalize so the mean is 1.0 (mean-1 normalization keeps the effective
    # epoch size equal to len(dataset), which matches HF Trainer's expectations).
    total = sample_weights.sum()
    if total <= 0:
        logger.warning("All sample weights are zero; returning uniform weights.")
        return np.ones(n, dtype=np.float64)
    sample_weights = sample_weights / total * n

    logger.info(
        f"Sampling strategy='{sampling_strategy}' weights: "
        f"min={sample_weights.min():.4g}, max={sample_weights.max():.4g}, "
        f"mean={sample_weights.mean():.4g}, "
        f"{'long-silence entries inflated' if long_silence_mask.any() else ''}"
    )
    return sample_weights


class DistributedWeightedSampler(Sampler):
    """
    Weighted random sampler compatible with distributed (DDP) training.
    Each rank samples a disjoint subset with the given per-sample weights.
    """

    def __init__(
        self,
        weights: np.ndarray,
        num_samples: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        import torch.distributed as dist

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.total_size = num_samples if num_samples is not None else len(weights)
        self.num_samples_per_rank = math.ceil(self.total_size / self.num_replicas)
        # Ensure total is evenly divisible by num_replicas
        self.total_size = self.num_samples_per_rank * self.num_replicas

        self.weights = torch.from_numpy(weights).double()

    def __iter__(self):
        # Deterministic RNG so all ranks agree on the global permutation
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Draw weighted samples (with replacement) for the full dataset
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()

        # Subsample for this rank
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples_per_rank
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Enhanced Trainer for fine-tuning (LLRD + EMA + BCE loss)
# ---------------------------------------------------------------------------


class NapeFineTuneTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_collator=None,
        ema_decay=0.9999,
        use_ema=True,
        base_lr=1e-4,
        head_lr=1e-3,
        llrd=0.75,
        llrd_end=None,
        weight_decay=0.05,
        mixup_fn=None,
        sample_weights=None,
        label_smoothing=0.0,
        task_type="multi_label",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None
        self.base_lr = base_lr
        self.head_lr = head_lr if head_lr is not None else base_lr
        self.use_llrd = llrd != 1.0  # If llrd is 1.0, no layer-wise decay
        self.llrd = llrd
        self.llrd_end = llrd_end
        self.weight_decay = weight_decay
        self.mixup_fn = mixup_fn
        self.sample_weights = sample_weights
        self.label_smoothing = label_smoothing
        if task_type not in ("multi_label", "single_label"):
            raise ValueError(f"task_type must be 'multi_label' or 'single_label', got {task_type!r}")
        self.task_type = task_type

    def _get_train_sampler(self, dataset=None):
        """Override to use balanced sampling when sample_weights are provided."""
        if self.sample_weights is None:
            return super()._get_train_sampler(dataset)

        ds = dataset if dataset is not None else self.train_dataset
        if self.args.world_size > 1:
            return DistributedWeightedSampler(
                weights=self.sample_weights,
                num_samples=len(ds),
                seed=self.args.seed,
            )
        else:
            return WeightedRandomSampler(
                weights=torch.from_numpy(self.sample_weights).double(),
                num_samples=len(ds),
                replacement=True,
            )

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [
            r"bias", r"layernorm", r"rmsnorm", r"layer_scale",
            r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)",
        ]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is not None:
            old_collator = self.data_collator
            self.data_collator = self.eval_collator
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = old_collator
            return dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)

    def create_optimizer(self):
        """Create optimizer with optional LLRD or simple head/backbone LR split."""
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.base_lr
        head_lr = self.head_lr
        weight_decay = self.weight_decay

        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)

        # Identify head and embedding params
        head_param_ids = set()
        if hasattr(self.model, "classifier"):
            head_param_ids.update(id(p) for p in self.model.classifier.parameters())

        final_ln_ids = set()
        if hasattr(self.model, "nape") and hasattr(self.model.nape, "layernorm"):
            final_ln_ids.update(id(p) for p in self.model.nape.layernorm.parameters())

        if self.use_llrd:
            # ---- LLRD mode: per-layer learning rates ----
            llrd = self.llrd

            embeddings_ids = set()
            if hasattr(self.model, "nape") and hasattr(self.model.nape, "embeddings"):
                embeddings_ids.update(id(p) for p in self.model.nape.embeddings.parameters())

            layers_param_ids = []
            if hasattr(self.model, "nape") and hasattr(self.model.nape, "encoder") and hasattr(self.model.nape.encoder, "layer"):
                encoder_layers = list(self.model.nape.encoder.layer)
                for blk in encoder_layers:
                    layers_param_ids.append(set(id(p) for p in blk.parameters()))
            else:
                encoder_layers = []
            num_layers = len(encoder_layers)

            grouped = defaultdict(list)

            for full_name, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue

                if id(p) in head_param_ids or id(p) in final_ln_ids:
                    lr_base = head_lr
                    scale = 0
                else:
                    assigned = False
                    for i in range(num_layers):
                        if id(p) in layers_param_ids[i]:
                            lr_base = base_lr
                            scale = (num_layers - 1 - i)
                            assigned = True
                            break
                    if not assigned:
                        if id(p) in embeddings_ids:
                            lr_base = base_lr
                            scale = num_layers
                        else:
                            lr_base = base_lr
                            scale = 0

                if p.ndim <= 1:
                    wd = 0.0
                else:
                    wd = weight_decay if full_name in decay_parameters else 0.0

                grouped[(lr_base, wd, scale)].append(p)

            optimizer_grouped_parameters = []
            for (lr_base, wd, scale), params in grouped.items():
                optimizer_grouped_parameters.append({
                    "params": params,
                    "lr": lr_base,
                    "weight_decay": wd,
                    "llrd": llrd,
                    "llrd_scale": scale,
                })

        else:
            # ---- No LLRD mode: simple head/backbone split ----
            head_decay, head_no_decay = [], []
            backbone_decay, backbone_no_decay = [], []

            for full_name, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue

                is_head = id(p) in head_param_ids or id(p) in final_ln_ids
                is_decay = (p.ndim > 1) and (full_name in decay_parameters)

                if is_head:
                    if is_decay:
                        head_decay.append(p)
                    else:
                        head_no_decay.append(p)
                else:
                    if is_decay:
                        backbone_decay.append(p)
                    else:
                        backbone_no_decay.append(p)

            optimizer_grouped_parameters = []
            if head_decay:
                optimizer_grouped_parameters.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
            if head_no_decay:
                optimizer_grouped_parameters.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})
            if backbone_decay:
                optimizer_grouped_parameters.append({"params": backbone_decay, "lr": base_lr, "weight_decay": weight_decay})
            if backbone_no_decay:
                optimizer_grouped_parameters.append({"params": backbone_no_decay, "lr": base_lr, "weight_decay": 0.0})

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = self.optimizer if optimizer is None else optimizer

        if self.use_llrd:
            # ---- LLRD mode: use custom LLRD scheduler ----
            scheduler_specific_kwargs = dict(self.args.lr_scheduler_kwargs or {})
            sched_name = scheduler_specific_kwargs.pop("custom_scheduler_type", None)

            if sched_name == "llrd_cosine_warmup":
                from schedulers import get_llrd_cosine_schedule_with_warmup
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                self.lr_scheduler = get_llrd_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    llrd_end=self.llrd_end,
                    **scheduler_specific_kwargs,
                )
                self._created_lr_scheduler = True

            elif sched_name == "llrd_cosine":
                from schedulers import get_llrd_cosine_schedule
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                self.lr_scheduler = get_llrd_cosine_schedule(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    llrd_end=self.llrd_end,
                    **scheduler_specific_kwargs,
                )
                self._created_lr_scheduler = True

        # If no custom scheduler was created (either no LLRD, or unrecognized sched_name),
        # fall through to HuggingFace's standard scheduler (cosine with warmup by default)
        # Clean up custom keys so HF doesn't choke on them
        if self.args.lr_scheduler_kwargs and "custom_scheduler_type" in self.args.lr_scheduler_kwargs:
            self.args.lr_scheduler_kwargs = {
                k: v for k, v in self.args.lr_scheduler_kwargs.items()
                if k != "custom_scheduler_type"
            }
        super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # Task-type dispatch.
        #   multi_label  → BCE. For AudioSet, mixup-mixed multi-hot labels still
        #                  go through sigmoid BCE, as in AST/EAT.
        #   single_label → CE with soft targets. For ESC-50 / GSC the labels are
        #                  (possibly mixup/cutmix soft) one-hot vectors that sum
        #                  to 1; CE with soft targets handles them natively:
        #                  L = -sum(target * log_softmax(logits)).
        if self.task_type == "single_label":
            # Symmetric label smoothing for single-label tasks. Unlike the
            # asymmetric AudioSet smoothing (which only softens positives so
            # the sigmoid outputs aren't pushed up across all 527 classes),
            # standard single-label smoothing redistributes ε uniformly across
            # non-target classes.
            if self.label_smoothing > 0 and self.model.training:
                K = labels.shape[-1]
                eps = self.label_smoothing
                labels = labels * (1.0 - eps) + eps / K

            # CE with soft targets. F.cross_entropy accepts probability
            # targets directly since torch >= 1.10, but the manual form makes
            # the mixup interaction obvious and avoids any dtype surprises.
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(labels * log_probs).sum(dim=-1).mean()
        else:
            # Multi-label (AudioSet). Asymmetric label smoothing: only the
            # positive labels are softened (1 → 1-ε), negatives stay at 0 so
            # the sigmoid outputs don't drift positive across all classes.
            if self.label_smoothing > 0 and self.model.training:
                labels = labels * (1.0 - self.label_smoothing)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss

    # --- EMA management ---

    def _init_ema_model(self):
        if self.ema_model is None:
            import copy
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            self.ema_model = self.ema_model.float()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema_model()
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    model_param = msd[k].float()
                    v.mul_(self.ema_decay).add_(model_param, alpha=1.0 - self.ema_decay)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > getattr(self, "_ema_global_step", 0):
            self._update_ema()
            self._ema_global_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix + "_ema", **gen_kwargs)
            self.model = backup
            out.update(ema_out)
        return out

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = f"{output_dir}/pytorch_model_ema.bin"
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)
            self.log({"ema_model_saved": ema_path})

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.use_ema:
            ema_ckpt = os.path.join(resume_from_checkpoint, "pytorch_model_ema.bin")
            if os.path.exists(ema_ckpt):
                if self.ema_model is None:
                    import copy
                    self.ema_model = copy.deepcopy(self.model)
                    for p in self.ema_model.parameters():
                        p.requires_grad_(False)
                state_dict = torch.load(ema_ckpt, map_location="cpu")
                missing, unexpected = self.ema_model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(f"[EMA] Missing keys: {missing}, Unexpected keys: {unexpected}")
            else:
                print(f"[EMA] No EMA checkpoint found at {ema_ckpt}, starting fresh EMA.")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class DataTrainingArguments:
    dataset_root: Optional[str] = field(
    default=None,
    metadata={"help": "Root directory prepended to every 'wav' path in the "
              "manifest. Manifests ship with paths relative to this root; "
              "set this to point at your local dataset location."},
    )
    train_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train manifest.json (from extract_wavs.py)."},
    )
    eval_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval/validation manifest.json. Used for per-epoch "
                  "evaluation during training and for model selection via "
                  "--load_best_model_at_end. For datasets with an explicit "
                  "validation/test split (GSC), point this at the validation set "
                  "and use --test_manifest for the test set."},
    )
    test_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test manifest.json. If set, after training "
                  "completes the best-validation model is evaluated ONCE on this "
                  "manifest and results are logged as eval_test_* and "
                  "eval_ema_test_*. Leave unset for datasets without a separate "
                  "test set (AudioSet, ESC-50 per-fold)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training examples for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate eval examples for debugging."},
    )
    norm_mean: float = field(
        default=-4.2677393,
        metadata={"help": "Mean for spectrogram normalization (AST default)."},
    )
    norm_std: float = field(
        default=4.5689974,
        metadata={"help": "Std for spectrogram normalization (AST default)."},
    )
    head_lr: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for classification head."},
    )
    # SpecAugment params
    freq_mask_param: int = field(
        default=48,
        metadata={"help": "Maximum frequency mask width for SpecAugment."},
    )
    time_mask_param: int = field(
        default=192,
        metadata={"help": "Maximum time mask width for SpecAugment."},
    )
    num_freq_masks: int = field(
        default=2,
        metadata={"help": "Number of frequency masks for SpecAugment."},
    )
    num_time_masks: int = field(
        default=2,
        metadata={"help": "Number of time masks for SpecAugment."},
    )
    # Mixup params
    mixup_alpha: float = field(
        default=0.5,
        metadata={"help": "Mixup alpha (0 = disabled)."},
    )
    mixup_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of applying mixup."},
    )
    no_spec_augment: bool = field(
        default=False,
        metadata={"help": "Disable SpecAugment during training."},
    )
    no_mixup: bool = field(
        default=False,
        metadata={"help": "Disable mixup during training."},
    )
    # CutMix params
    cutmix_alpha: float = field(
        default=0.0,
        metadata={"help": "CutMix alpha (0 = disabled). Recommended: 1.0."},
    )
    cutmix_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of applying CutMix per batch."},
    )
    # Audio rolling and noise (AST/AudioMAE/EAT-style)
    use_audio_rolling: bool = field(
        default=False,
        metadata={"help": "Enable random circular time-shift on spectrogram (AST/EAT-style)."},
    )
    use_random_noise: bool = field(
        default=False,
        metadata={"help": "Enable additive random noise on spectrogram (AST/EAT-style)."},
    )
    noise_scale: float = field(
        default=0.1,
        metadata={"help": "Scale factor for random noise augmentation. "
                  "Following AST/EAT: noise = rand() * rand() * noise_scale."},
    )
    # Balanced sampling (for AS-2M / unbalanced sets)
    balanced_sampling: bool = field(
        default=False,
        metadata={"help": "Use inverse-frequency weighted sampling to oversample rare classes. "
                  "Recommended for fine-tuning on unbalanced AudioSet (AS-2M). "
                  "Not needed for balanced sets (AS-20K)."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing value. 0.0 = disabled. "
                  "Multi-label: asymmetric (positive labels only smoothed 1 → 1-ε). "
                  "Single-label: symmetric (mass ε redistributed uniformly over non-target classes). "
                  "Recommended range: 0.05–0.1."},
    )
    # -----------------------------------------------------------------
    # Task & dataset flexibility (added for ESC-50 and GSC fine-tuning)
    # -----------------------------------------------------------------
    task_type: str = field(
        default="multi_label",
        metadata={"help": "Classification task type. 'multi_label' uses BCE + mAP "
                  "(AudioSet). 'single_label' uses CE-with-soft-targets + top-1 "
                  "accuracy (ESC-50, GSC). Mixup/cutmix work for both."},
    )
    audio_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Override pretraining audio_duration (seconds). "
                  "If None, uses the value from the model config (10.0 for AudioSet "
                  "pretraining). Set to 5.0 for ESC-50, 1.0 for GSC. When overridden, "
                  "target_time_frames is automatically recomputed (ceil to patch_size multiple)."},
    )
    fold_held_out: int = field(
        default=0,
        metadata={"help": "For fold-based cross-validation (ESC-50). Set to 1..5 to "
                  "hold out that fold as eval and train on the rest. 0 (default) = "
                  "no fold filtering (AudioSet / GSC have explicit train/eval splits)."},
    )
    classes_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to classes.json (produced by extract_esc50.py / "
                  "extract_gsc.py / make_gsc_v1_ks1.py). If set, loads the class "
                  "list from this file rather than auto-discovering from the "
                  "manifest. Required for ESC-50 (stable class ids across folds) "
                  "and recommended for GSC. AudioSet can leave this unset."},
    )
    sampling_strategy: str = field(
        default="none",
        metadata={"help": "Per-sample sampling weights for the train sampler. "
                  "'none' = uniform random. 'inverse_frequency' = AudioSet-style "
                  "(every class equal mass). 'uniform_balanced' = same as "
                  "inverse_frequency for single-label but explicit in naming. "
                  "'beats_style' = BEATs KS1 rule (non-unknown classes each get "
                  "50% of the unknown class's mass per epoch). The existing "
                  "--balanced_sampling flag, if True, is treated as "
                  "--sampling_strategy inverse_frequency for backward compat."},
    )
    beats_unknown_class: str = field(
        default="_unknown_",
        metadata={"help": "For --sampling_strategy beats_style: the label string "
                  "that plays the role of 'unknown' in BEATs' rule. "
                  "Default '_unknown_' matches SUPERB KS1 convention."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to finetuning-ready model (from init_nape_cls_from_pretrain.py)."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Freeze entire nape backbone (linear probe)."},
    )
    freeze_embed: bool = field(
        default=False,
        metadata={"help": "Freeze patch embedding layer."},
    )
    use_ema: bool = field(
        default=True,
        metadata={"help": "Use EMA model."},
    )
    ema_decay: float = field(
        default=0.9999,
        metadata={"help": "EMA decay rate."},
    )
    use_llrd: bool = field(
        default=True,
        metadata={"help": "Use layer-wise learning rate decay. If False, all layers use the same LR."},
    )
    llrd: float = field(
        default=0.65,
        metadata={"help": "Layer-wise learning rate decay factor (only used if use_llrd=True)."},
    )
    llrd_end: Optional[float] = field(
        default=None,
        metadata={"help": "LLRD at end of training (for annealing)."},
    )
    drop_path_prob: float = field(
        default=0.0,
        metadata={"help": "Drop path (stochastic depth) rate during fine-tuning. EAT/MAE use 0.1 for base."},
    )
    bidirectional: bool = field(
        default=True,
        metadata={"help": "Use bidirectional (non-causal) attention for fine-tuning. "
                  "If False, keeps causal attention from pretraining."},
    )
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "Pooling mode for classification. "
                  "'mean' = mean-pool over all patch tokens (best with bidirectional attention). "
                  "'last_token' = use last sequence token (best with causal attention, as in NEPA). "
                  "'cls_token' = use CLS token output (best when pretrained with CLS objective). "
                  "'attentive' = learnable cross-attention pooling over patch tokens (V-JEPA style)."},
    )
    num_probe_queries: int = field(
        default=1,
        metadata={"help": "Number of learnable query tokens for attentive probing. "
                  "Only used when pooling_mode='attentive'. Typical values: 1-16."},
    )
    probe_layer: Optional[str] = field(
        default=None,
        metadata={"help": "Which layer to feed the classifier from. Default None → "
                  "use the last transformer block (standard fine-tuning behavior). "
                  "Options: 'last' (= last block), an integer K in [0, num_hidden_layers-1] "
                  "(the K-th transformer block), or 'weighted_sum' (learned softmax weighting "
                  "over all L+1 hidden states, SUPERB-style). Typically used for linear "
                  "probing (combine with --freeze_backbone True)."},
    )
    num_register_tokens: int = field(
        default=0,
        metadata={"help": "Number of learnable register tokens to add during fine-tuning. "
                  "Registers are placed after CLS and before patches, receive no positional "
                  "encoding, and act as global information aggregation slots. Default: 0 (disabled). "
                  "Typical values: 4 or 8."},
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_map(eval_pred):
    """Compute mean average precision for multi-label classification."""
    logits, labels = eval_pred
    # Sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Safety check: replace NaN with 0 (can happen with newly initialized modules)
    if np.any(np.isnan(probs)):
        logger.warning("NaN detected in evaluation probabilities — replacing with 0")
        probs = np.nan_to_num(probs, nan=0.0)

    # Per-class AP, then average
    # Skip classes with no positive examples in this eval set
    aps = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            ap = average_precision_score(labels[:, i], probs[:, i])
            aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0
    return {"mAP": mAP}


def compute_accuracy(eval_pred):
    """Compute top-1 accuracy for single-label classification.

    Labels arrive as multi-hot with (at eval time) exactly one 1 per row —
    we argmax both logits and labels to get the predicted/true class index.
    """
    logits, labels = eval_pred

    if np.any(np.isnan(logits)):
        logger.warning("NaN detected in evaluation logits — replacing with 0")
        logits = np.nan_to_num(logits, nan=0.0)

    preds = np.argmax(logits, axis=-1)
    truth = np.argmax(labels, axis=-1)
    acc = float((preds == truth).mean())
    return {"accuracy": acc}


def get_metrics_fn(task_type: str):
    """Return the appropriate compute_metrics function for the task type."""
    if task_type == "multi_label":
        return compute_map
    elif task_type == "single_label":
        return compute_accuracy
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---- Logging ----
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}"
    )

    # ---- Checkpoint detection ----
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # ---- Build label map ----
    # Priority:
    #   1) If --classes_file is given, use its frozen class list (recommended
    #      for ESC-50 and GSC so label ids stay stable across folds/splits).
    #   2) Otherwise, auto-discover labels from the manifest (AudioSet default).
    if data_args.train_manifest is None and data_args.eval_manifest is None:
        raise ValueError("Must provide at least --train_manifest or --eval_manifest.")

    if data_args.classes_file is not None:
        label2id, id2label = load_label_map_from_classes_file(data_args.classes_file)
    else:
        manifest_for_labels = data_args.train_manifest or data_args.eval_manifest
        label2id, id2label = build_audioset_label_map(manifest_for_labels)
        logger.info(f"Built label map from manifest: {len(label2id)} labels from {manifest_for_labels}")
    num_labels = len(label2id)
    logger.info(f"task_type={data_args.task_type}, num_labels={num_labels}")

    # ---- Model & config ----
    config = NapeConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.num_labels = num_labels
    config.is_causal = not model_args.bidirectional
    config.output_hidden_states = False  # disable; only needed during pretraining with deep supervision
    if model_args.bidirectional:
        logger.info("Bidirectional (non-causal) attention enabled for fine-tuning")
    else:
        logger.info("Causal attention kept for fine-tuning")

    # ---- Optional audio_duration override ----
    # If --audio_duration is set (e.g. 5.0 for ESC-50, 1.0 for GSC), recompute
    # target_time_frames accordingly: ceil(duration * sr / hop) rounded UP to
    # the nearest multiple of patch_size. RoPE extrapolates to the new grid
    # automatically (verified empirically via verify_rope_variable_duration.py).
    if data_args.audio_duration is not None:
        new_duration = float(data_args.audio_duration)
        if abs(new_duration - config.audio_duration) > 1e-6:
            raw_frames = round(new_duration * config.sample_rate / config.hop_length)
            ps = config.patch_size
            new_target = ((raw_frames + ps - 1) // ps) * ps
            logger.info(
                f"Audio duration override: {config.audio_duration}s -> {new_duration}s "
                f"(target_time_frames {config.target_time_frames} -> {new_target}, "
                f"num_time_patches {config.target_time_frames // ps} -> {new_target // ps})"
            )
            config.audio_duration = new_duration
            config.target_time_frames = new_target
        else:
            logger.info(f"Audio duration unchanged ({config.audio_duration}s).")

    # Set pooling mode
    if model_args.pooling_mode in ("mean", "last_token", "cls_token", "attentive"):
        config.pooling_mode = model_args.pooling_mode
        config.add_pooling_layer = model_args.pooling_mode in ("mean", "cls_token", "attentive")
        logger.info(f"Pooling mode: {model_args.pooling_mode}")
        if model_args.pooling_mode == "attentive":
            config.num_probe_queries = model_args.num_probe_queries
            logger.info(f"Attentive probing with {model_args.num_probe_queries} query token(s)")
    else:
        raise ValueError(f"Unknown pooling_mode: {model_args.pooling_mode}. Must be 'mean', 'last_token', 'cls_token', or 'attentive'.")

    # Set register tokens (fine-tuning only, not present in pretrained model)
    if model_args.num_register_tokens > 0:
        config.num_register_tokens = model_args.num_register_tokens
        logger.info(f"Register tokens: {model_args.num_register_tokens} (fine-tuning only, randomly initialized)")

    if model_args.drop_path_prob > 0:
        config.drop_path_prob = model_args.drop_path_prob
        logger.info(f"Drop path enabled: {model_args.drop_path_prob}")

    # Probe layer: parse the CLI string into the value expected by
    # NapeForClassification. None (default) → normal fine-tuning behavior
    # (use last transformer block). "last" / int / "weighted_sum" all supported.
    if model_args.probe_layer is not None and model_args.probe_layer != "":
        raw = model_args.probe_layer
        if raw == "last":
            config.probe_layer = "last"
        elif raw == "weighted_sum":
            config.probe_layer = "weighted_sum"
        else:
            try:
                config.probe_layer = int(raw)
            except ValueError:
                raise ValueError(
                    f"--probe_layer must be 'last', 'weighted_sum', or an integer. "
                    f"Got: {raw!r}"
                )
        logger.info(f"Linear probing: probe_layer={config.probe_layer}")

    model = NapeForClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True,  # classifier head may differ
    )

    # Reinitialize register tokens after loading pretrained weights
    # (they don't exist in the pretrained checkpoint and may be zeroed by from_pretrained)
    if model_args.num_register_tokens > 0:
        embeddings = model.nape.embeddings
        if hasattr(embeddings, 'register_tokens'):
            nn.init.normal_(embeddings.register_tokens, mean=0.0, std=0.02)
            logger.info(f"Reinitialized {model_args.num_register_tokens} register tokens (std=0.02)")

    # Reinitialize attentive probe after loading pretrained weights
    # (it doesn't exist in the pretrained checkpoint)
    if model_args.pooling_mode == "attentive" and hasattr(model, 'attentive_probe'):
        nn.init.normal_(model.attentive_probe.query_tokens, mean=0.0, std=0.02)
        for m in model.attentive_probe.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        logger.info(f"Reinitialized attentive probe ({model_args.num_probe_queries} queries)")

    # Freeze options
    if model_args.freeze_backbone:
        for param in model.nape.parameters():
            param.requires_grad = False
        logger.info("Froze entire nape backbone (linear probe mode)")

    if model_args.freeze_embed and not model_args.freeze_backbone:
        for param in model.nape.embeddings.patch_embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze patch embedding layer")

    # Handle LLRD flag
    effective_llrd = model_args.llrd if model_args.use_llrd else 1.0
    if not model_args.use_llrd:
        logger.info("LLRD disabled — all layers use the same learning rate")

    # Log trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # ---- Spectrogram extractor ----
    spec_extractor = SpectrogramExtractor(
        config,
        norm_mean=data_args.norm_mean,
        norm_std=data_args.norm_std,
    )

    # ---- Augmentations ----
    train_spec_augment = None
    if not data_args.no_spec_augment:
        train_spec_augment = SpecAugment(
            freq_mask_param=data_args.freq_mask_param,
            time_mask_param=data_args.time_mask_param,
            num_freq_masks=data_args.num_freq_masks,
            num_time_masks=data_args.num_time_masks,
        )
        logger.info(f"SpecAugment enabled: freq_mask={data_args.freq_mask_param}, "
                     f"time_mask={data_args.time_mask_param}")

    mixup_fn = None
    if not data_args.no_mixup and data_args.mixup_alpha > 0:
        mixup_fn = SpectrogramMixup(
            alpha=data_args.mixup_alpha,
            prob=data_args.mixup_prob,
        )
        logger.info(f"Mixup enabled: alpha={data_args.mixup_alpha}, prob={data_args.mixup_prob}")

    cutmix_fn = None
    if data_args.cutmix_alpha > 0:
        cutmix_fn = SpectrogramCutMix(
            alpha=data_args.cutmix_alpha,
            prob=data_args.cutmix_prob,
        )
        logger.info(f"CutMix enabled: alpha={data_args.cutmix_alpha}, prob={data_args.cutmix_prob}")

    # Per-sample augmentations (applied in dataset __getitem__)
    train_audio_rolling = None
    if data_args.use_audio_rolling:
        train_audio_rolling = AudioRolling()
        logger.info("Audio rolling (circular time shift) enabled")

    train_random_noise = None
    if data_args.use_random_noise:
        train_random_noise = RandomNoise(noise_scale=data_args.noise_scale)
        logger.info(f"Random noise enabled: scale={data_args.noise_scale}")

    if data_args.label_smoothing > 0:
        logger.info(f"Asymmetric label smoothing enabled: ε={data_args.label_smoothing} "
                     f"(positive labels: 1 → {1.0 - data_args.label_smoothing})")

    # ---- Datasets ----
    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        train_dataset = WavManifestClassificationDataset(
            data_args.train_manifest,
            spec_extractor,
            label2id,
            spec_augment=train_spec_augment,
            audio_rolling=train_audio_rolling,
            random_noise=train_random_noise,
            max_samples=data_args.max_train_samples,
            fold_held_out=data_args.fold_held_out,
            mode="train",
            dataset_root=data_args.dataset_root,
        )

    if training_args.do_eval:
        eval_dataset = WavManifestClassificationDataset(
            data_args.eval_manifest,
            spec_extractor,
            label2id,
            spec_augment=None,  # No augmentation for eval
            max_samples=data_args.max_eval_samples,
            fold_held_out=data_args.fold_held_out,
            mode="eval",
            dataset_root=data_args.dataset_root,
        )

    # ---- Collate functions ----
    def train_collate_fn(examples):
        spectrograms = torch.stack([ex["spectrogram"] for ex in examples])
        labels = torch.stack([ex["labels"] for ex in examples])

        # Apply mixup in collate (on batched data)
        if mixup_fn is not None:
            spectrograms, labels = mixup_fn(spectrograms, labels)

        # Apply CutMix in collate (on batched data)
        if cutmix_fn is not None:
            spectrograms, labels = cutmix_fn(spectrograms, labels)

        return {"spectrogram": spectrograms, "labels": labels}

    def eval_collate_fn(examples):
        spectrograms = torch.stack([ex["spectrogram"] for ex in examples])
        labels = torch.stack([ex["labels"] for ex in examples])
        return {"spectrogram": spectrograms, "labels": labels}

    # ---- Sampling weights ----
    # Resolution order for the sampling strategy:
    #   1. Explicit --sampling_strategy if not "none" (new-style flag).
    #   2. --balanced_sampling True (legacy) → treat as "inverse_frequency"
    #      for backward compatibility with existing AudioSet recipes.
    #   3. Otherwise no sampler (uniform random).
    resolved_strategy = data_args.sampling_strategy
    if resolved_strategy == "none" and data_args.balanced_sampling:
        resolved_strategy = "inverse_frequency"
        logger.info("Legacy --balanced_sampling True detected; using sampling_strategy=inverse_frequency.")

    sample_weights = None
    if resolved_strategy != "none" and data_args.train_manifest:
        sample_weights = compute_sample_weights(
            data_args.train_manifest,
            label2id,
            sampling_strategy=resolved_strategy,
            beats_unknown_class=data_args.beats_unknown_class,
            fold_held_out=data_args.fold_held_out,
        )
        logger.info(f"Weighted sampling enabled (strategy={resolved_strategy}).")

    # ---- Trainer ----
    trainer = NapeFineTuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_metrics_fn(data_args.task_type),
        data_collator=train_collate_fn,
        eval_collator=eval_collate_fn,
        ema_decay=model_args.ema_decay,
        use_ema=model_args.use_ema,
        base_lr=training_args.learning_rate,
        head_lr=data_args.head_lr,
        llrd=effective_llrd,
        llrd_end=model_args.llrd_end,
        weight_decay=training_args.weight_decay,
        mixup_fn=mixup_fn,
        sample_weights=sample_weights,
        label_smoothing=data_args.label_smoothing,
        task_type=data_args.task_type,
    )

    # ---- Training ----
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ---- Evaluation on validation set ----
    # After training with --load_best_model_at_end, the trainer now holds the
    # checkpoint that scored best on the validation set. We log that validation
    # score here as eval_*.
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ---- Final test-set evaluation ----
    # For datasets with a separate test split (GSC v1 / v2), evaluate the
    # best-validation model ONCE on the test set. Logs appear as test_* and
    # EMA variant test_ema_* so they don't collide with the per-epoch eval_*
    # logs in wandb.
    if data_args.test_manifest:
        logger.info(f"Running final test evaluation on {data_args.test_manifest}")
        test_dataset = WavManifestClassificationDataset(
            data_args.test_manifest,
            spec_extractor,
            label2id,
            spec_augment=None,  # no augmentation at test time
            max_samples=None,
            fold_held_out=0,    # test set is never fold-filtered
            mode="eval",
            dataset_root=data_args.dataset_root,
        )
        
        # NOTE: we deliberately do NOT call trainer.evaluate(...) here.
        # HF Trainer's get_eval_dataloader caches the eval dataloader when
        # --dataloader_persistent_workers is True, keyed as "eval". After the
        # in-training validation runs, that cache holds the VAL dataloader.
        # Calling trainer.evaluate(eval_dataset=test_dataset, ...) then hits
        # the cache and silently re-evaluates on the val set, producing
        # identical val and test metrics.
        #
        # Workaround: build a DataLoader from the test dataset by hand and
        # drive trainer.evaluation_loop directly. evaluation_loop takes a
        # dataloader as its argument with no cache lookup, so it's immune
        # to the collision above.
        from torch.utils.data import DataLoader
 
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.eval_batch_size,
            collate_fn=trainer.eval_collator or trainer.data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            shuffle=False,
        )
        output = trainer.evaluation_loop(
            test_dataloader,
            description="Test Evaluation",
            prediction_loss_only=False,
            ignore_keys=None,
            metric_key_prefix="test",
        )
        
        test_metrics = dict(output.metrics)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()