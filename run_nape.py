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
"""Pre-training script for Nape — next-embedding prediction on audio spectrograms."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names

from models import NapeForPreTraining, NapeConfig

import warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrogram extraction
# ---------------------------------------------------------------------------

class SpectrogramExtractor:
    """Compute log-mel fbank features using Kaldi-compatible extraction (matches AST/SSAST/EAT)."""

    def __init__(self, config: NapeConfig, norm_mean: float = -4.2677393, norm_std: float = 4.5689974):
        self.sample_rate = config.sample_rate
        self.target_time_frames = config.target_time_frames
        self.freq_bins = config.freq_bins
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.target_samples = int(config.audio_duration * config.sample_rate)

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Args:
            waveform: [num_channels, num_samples] or [num_samples]
            sr: original sample rate

        Returns:
            spectrogram: [1, freq_bins, target_time_frames]
        """
        # Ensure mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Pad or truncate to target duration
        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            waveform = F.pad(waveform, (0, self.target_samples - num_samples))
        elif num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]

        # Extract fbank using Kaldi-compatible method (matches AST/SSAST/EAT)
        # Input: [channels, samples], Output: [time_frames, num_mel_bins]
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

        # Normalize (AST standard: mean=0, std=0.5)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # Transpose to [F, T] then add channel dim: [1, F, T]
        fbank = fbank.T.unsqueeze(0)

        # Pad or truncate time frames to target
        time_frames = fbank.shape[-1]
        if time_frames < self.target_time_frames:
            fbank = F.pad(fbank, (0, self.target_time_frames - time_frames))
        elif time_frames > self.target_time_frames:
            fbank = fbank[:, :, :self.target_time_frames]

        return fbank  # [1, freq_bins, target_time_frames]


# ---------------------------------------------------------------------------
# Wav Manifest Dataset — loads pre-extracted wav files (matches AST/SSAST/EAT)
# ---------------------------------------------------------------------------

class WavManifestDataset(Dataset):
    """
    PyTorch Dataset that loads audio from pre-extracted wav files via a manifest JSON.
    Computes fbank spectrograms on-the-fly using torchaudio.

    Expected manifest format (from extract_wavs.py):
        {"data": [{"wav": "/path/to/file.wav", "video_id": "...", ...}, ...]}
    """

    def __init__(self, manifest_path: str, spec_extractor: SpectrogramExtractor,
                 max_samples: Optional[int] = None, dataset_root: str = None):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.entries = manifest["data"]
        if max_samples is not None:
            self.entries = self.entries[:max_samples]

        self.spec_extractor = spec_extractor
        self.dataset_root = dataset_root
        logger.info(f"Loaded wav manifest: {manifest_path} ({len(self.entries)} samples)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav_path = os.path.join(self.dataset_root, entry["wav"])

        # Load wav file — already resampled to 16kHz by extract_wavs.py
        waveform, sr = torchaudio.load(wav_path)

        # Compute spectrogram
        spectrogram = self.spec_extractor(waveform, sr)

        return {"spectrogram": spectrogram}


# ---------------------------------------------------------------------------
# Enhanced Trainer (EMA + separate embed LR)
# ---------------------------------------------------------------------------

class EnhancedTrainer(Trainer):
    def __init__(self, *args, embed_lr=None, ema_decay=0.9999, use_ema=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [
            r"bias", r"layernorm", r"rmsnorm", r"layer_scale",
            r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)",
        ]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.embed_lr is None:
            return super().create_optimizer()

        decay_params = set(self.get_decay_parameter_names(self.model))
        embed_params = set(
            f"nape.embeddings.{n}"
            for n, _ in self.model.nape.embeddings.named_parameters()
        )

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        groups = [
            {"params": [], "weight_decay": wd,  "lr": self.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},
            {"params": [], "weight_decay": wd,  "lr": base_lr},
            {"params": [], "weight_decay": 0.0, "lr": base_lr},
        ]

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_decay = name in decay_params
            is_embed = name in embed_params

            if is_embed and is_decay:
                groups[0]["params"].append(p)
            elif is_embed and not is_decay:
                groups[1]["params"].append(p)
            elif not is_embed and is_decay:
                groups[2]["params"].append(p)
            else:
                groups[3]["params"].append(p)

        optimizer_grouped_parameters = [g for g in groups if g["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

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
                if k not in msd:
                    continue
                # Skip non-float buffers (e.g. BatchNorm's num_batches_tracked is Long).
                # For these, just copy the model's value through — they're counters,
                # not parameters that benefit from EMA smoothing.
                if not v.is_floating_point():
                    v.copy_(msd[k])
                    continue
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        # Log individual loss components on logging-step boundaries
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {}
            if log_dict:
                self.log(log_dict)

        return (loss, outputs) if return_outputs else loss


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
        metadata={"help": "Path to train manifest.json from extract_wavs.py. "
                  "If provided, wav files are loaded directly (fast)."},
    )
    eval_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval/test manifest.json from extract_wavs.py."},
    )

    # --- HuggingFace datasets mode (fallback, slower) ---
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a dataset from the hub or path to a local dataset. "
                  "Only used if train_manifest is not provided."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Configuration name of the dataset."},
    )
    audioset_split: str = field(
        default="unbalanced",
        metadata={
            "help": "Which AudioSet split to use: 'balanced' or 'unbalanced'.",
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data."},
    )
    load_from_disk: bool = field(
        default=False,
        metadata={"help": "Whether to load dataset from disk using load_from_disk."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )

    # --- Common ---
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples for debugging."},
    )
    train_val_split: Optional[float] = field(
        default=None,
        metadata={"help": "Percent to split off of train for validation."},
    )
    norm_mean: float = field(
        default=-4.2677393,
        metadata={"help": "Mean for spectrogram normalization (AST default)."},
    )
    norm_std: float = field(
        default=4.5689974,
        metadata={"help": "Std for spectrogram normalization (AST default)."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from the hub."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace auth token."},
    )
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for the embedding layer."},
    )


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
    logger.info(f"Training/evaluation parameters {training_args}")

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

    # ---- Model & config ----
    config = NapeConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.model_name_or_path:
        model = NapeForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        logger.info("Training new model from scratch")
        model = NapeForPreTraining(config)

    # ---- Spectrogram extractor ----
    spec_extractor = SpectrogramExtractor(
        config,
        norm_mean=data_args.norm_mean,
        norm_std=data_args.norm_std,
    )

    # ---- Collate function (shared by both data loading modes) ----
    def collate_fn(examples):
        spectrograms = torch.stack([example["spectrogram"] for example in examples])
        return {"spectrogram": spectrograms}

    # ---- Load data ----
    train_dataset = None
    eval_dataset = None

    logger.info("Using wav manifest mode")
    logger.info(f"  Train manifest: {data_args.train_manifest}")

    if training_args.do_train:
        train_dataset = WavManifestDataset(
            data_args.train_manifest,
            spec_extractor,
            max_samples=data_args.max_train_samples,
            dataset_root=data_args.dataset_root,
        )

    if training_args.do_eval:
        if data_args.eval_manifest is None:
            raise ValueError("--do_eval requires --eval_manifest when using wav manifest mode.")
        eval_dataset = WavManifestDataset(
            data_args.eval_manifest,
            spec_extractor,
            dataset_root=data_args.dataset_root,
        )

    # ---- Trainer ----
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
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

    # ---- Evaluation ----
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ---- Push to hub ----
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "audio-embedded-prediction",
        "dataset": data_args.dataset_name or data_args.train_manifest,
        "tags": ["audio", "embedded-prediction", "self-supervised"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
