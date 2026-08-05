#!/usr/bin/env python
"""
analyze_pretrain.py
===================

Intrinsic analysis of what Nape learns during pretraining. Operates on a
pretrained checkpoint without any downstream fine-tuning.

For each input clip, computes the per-patch cosine similarity between the
predicted next-patch embedding and the actual next-patch embedding. Produces:

  - One per-clip figure (up to --per_clip_limit clips): two stacked panels,
    log-mel spectrogram on top, per-patch cosine similarity grid on the
    bottom (aligned time axis).
  - One averaged figure across all clips: mean and standard deviation of the
    per-patch cosine similarity grid, side by side. Reveals systematic
    structure (e.g. weaker prediction at the start of each frequency row,
    in the padding column, or at the top frequencies) that per-clip noise
    otherwise hides.

Usage:
    python analyze_pretrain.py \\
        --checkpoint /abs/path/to/checkpoint \\
        --manifest   /abs/path/to/manifest.json \\
        --output_dir analysis_outputs \\
        --num_clips        500 \\
        --per_clip_limit   10
"""

import argparse
import json
import os
import random
import sys
import importlib
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# ----------------------------------------------------------------------------
# Import the model. modeling_nape.py uses relative imports
# (`from .configuration_nape import ...`), which only work when loaded
# as part of a package. We add the parent of this file's directory to sys.path
# and use importlib so the script runs both via `python analyze_pretrain.py`
# and `python -m nape.analyze_pretrain`.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_PKG = os.path.basename(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_modeling = importlib.import_module(f"{_PKG}.modeling_nape")
NapeForPreTraining = _modeling.NapeForPreTraining

# ----------------------------------------------------------------------------
# Audio -> spectrogram (mirrors the model's SpectrogramExtractor)
# ----------------------------------------------------------------------------

def extract_log_mel(
    wav_path: str,
    sample_rate: int = 16000,
    num_mel_bins: int = 128,
    target_length: int = 1008,
    norm_mean: float = -4.2677393,
    norm_std: float = 4.5689974,
    audio_duration: float = 10.0,
) -> torch.Tensor:
    """Load a wav and return a normalized log-mel spectrogram [num_mel_bins, target_length]."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_samples = int(audio_duration * sample_rate)
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))

    waveform = waveform - waveform.mean()

    fbank = kaldi.fbank(
        waveform,
        htk_compat=True, sample_frequency=sample_rate, use_energy=False,
        window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0,
        frame_shift=10, frame_length=25,
    )

    n_frames = fbank.shape[0]
    if n_frames > target_length:
        fbank = fbank[:target_length, :]
    elif n_frames < target_length:
        fbank = F.pad(fbank, (0, 0, 0, target_length - n_frames))

    fbank = fbank.transpose(0, 1)
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank


# ----------------------------------------------------------------------------
# Forward pass: extract z (targets) and z_hat (predictions)
# ----------------------------------------------------------------------------

def forward_clip(
    model: NapeForPreTraining,
    spectrogram: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single spectrogram through the pretrained model and extract the
    target patch embeddings z and the predicted next-patch embeddings z_hat.
    Both shapes: [N, D] where N = T_F * T_T is the number of patches.
    """
    model.eval()
    x = spectrogram.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.nape(x)
        z_input  = outputs.input_embedding
        h_output = outputs.last_hidden_state

        if getattr(model, 'use_prediction_head', False) and hasattr(model, 'prediction_head'):
            h_output = model.prediction_head(h_output)

        z_hat = h_output[:, :-1, :]
        z     = z_input [:, 1:,  :]

    return z_hat[0].float().cpu(), z[0].float().cpu()


# ----------------------------------------------------------------------------
# Compute the per-patch cosine similarity grid
# ----------------------------------------------------------------------------

def compute_prediction_quality_grid(
    z_hat: torch.Tensor,
    z: torch.Tensor,
    T_F: int,
    T_T: int,
    patch_order_perm: torch.Tensor = None,
) -> np.ndarray:
    """Return a [T_F, T_T] grid of per-patch cosine similarities, in raster order."""
    cos = F.cosine_similarity(z_hat, z, dim=-1).numpy()

    expected = T_F * T_T
    if cos.shape[0] >= expected:
        cos = cos[:expected]
    else:
        cos = np.pad(cos, (0, expected - cos.shape[0]), constant_values=np.nan)

    if patch_order_perm is not None:
        perm = patch_order_perm.numpy() if torch.is_tensor(patch_order_perm) else np.asarray(patch_order_perm)
        cos_raster = np.empty_like(cos)
        cos_raster[perm] = cos
        cos = cos_raster

    return cos.reshape(T_F, T_T)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_prediction_heatmap(
    spectrogram: torch.Tensor,
    cos_grid: np.ndarray,
    T_F: int,
    T_T: int,
    patch_size: int,
    output_path: str,
    title: str = None,
    vmin: float = 0.5,
    vmax: float = 1.0,
):
    """Stacked panels: spectrogram on top, per-patch cosine similarity below, sharex'd."""
    fig, (ax_spec, ax_cos) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    ax_spec.imshow(
        spectrogram.numpy(), aspect='auto', origin='lower',
        cmap='viridis', interpolation='none',
    )
    ax_spec.set_ylabel('Mel bin', fontsize = 18)
    if title:
        ax_spec.set_title(title, fontsize = 21)

    im = ax_cos.imshow(
        cos_grid, aspect='auto', origin='lower',
        cmap='magma', interpolation='none',
        extent=[0, T_T * patch_size, 0, T_F * patch_size],
        vmin=vmin, vmax=vmax,
    )
    ax_cos.set_xlabel('Time frame', fontsize = 18)
    ax_cos.set_ylabel('Patch grid', fontsize = 18)
    ax_cos.tick_params(axis='y', labelsize=15)
    ax_cos.tick_params(axis='x', labelsize=15)

    cbar = fig.colorbar(im, ax=[ax_spec, ax_cos], orientation='vertical',
                        fraction=0.025, pad=0.02)
    cbar.set_label('Cosine similarity', fontsize = 16)
    cbar.ax.tick_params(labelsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_averaged_heatmap(
    cos_grids: np.ndarray,
    T_F: int,
    T_T: int,
    patch_size: int,
    output_path: str,
    title: str = None,
    vmin: float = 0.5,
    vmax: float = 1.0,
):
    """
    Two-panel figure: mean and standard deviation of cosine similarity per
    patch position, computed across a batch of clips.
    """
    mean_grid = np.nanmean(cos_grids, axis=0)
    std_grid  = np.nanstd (cos_grids, axis=0)

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(16, 4))

    im_mean = ax_mean.imshow(
        mean_grid, aspect='auto', origin='lower',
        cmap='magma', interpolation='none',
        extent=[0, T_T * patch_size, 0, T_F * patch_size],
        vmin=vmin, vmax=vmax,
    )
    ax_mean.set_xlabel('Time frame', fontsize = 18)
    ax_mean.set_ylabel('Mel bin (patch grid)', fontsize = 18)
    ax_mean.tick_params(axis='y', labelsize=15)
    ax_mean.tick_params(axis='x', labelsize=15)
    #ax_mean.set_title(f'Mean prediction quality across {K} clips')
    plt.colorbar(im_mean, ax=ax_mean,  pad=0.02)

    im_std = ax_std.imshow(
        std_grid, aspect='auto', origin='lower',
        cmap='viridis', interpolation='none',
        extent=[0, T_T * patch_size, 0, T_F * patch_size],
    )
    ax_std.set_xlabel('Time frame', fontsize = 10)
    ax_std.set_ylabel('Mel bin (patch grid)', fontsize = 10)
    ax_std.tick_params(axis='y', labelsize=15)
    ax_std.tick_params(axis='x', labelsize=15)
    #ax_std.set_title(f'Std. of prediction quality across {K} clips')
    plt.colorbar(im_std, ax=ax_std,  pad=0.02)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.01)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--manifest',    required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--num_clips',   type=int, default=5)
    parser.add_argument('--per_clip_limit', type=int, default=10,
                        help='Maximum number of per-clip heatmaps to save. '
                             'The averaged figure is always saved when num_clips > 1.')

    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--sample_rate',    type=int, default=16000)
    parser.add_argument('--num_mel_bins',   type=int, default=128)
    parser.add_argument('--target_length',  type=int, default=1008)
    parser.add_argument('--norm_mean',      type=float, default=-4.2677393)
    parser.add_argument('--norm_std',       type=float, default=4.5689974)
    parser.add_argument('--audio_duration', type=float, default=10.0)
    parser.add_argument('--patch_size',     type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    T_F = args.num_mel_bins  // args.patch_size
    T_T = args.target_length // args.patch_size
    print(f"Patch grid: {T_F} (freq) x {T_T} (time) = {T_F * T_T} patches per clip")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint} (device={device})...")
    model = NapeForPreTraining.from_pretrained(args.checkpoint).to(device)
    model.eval()

    patch_order      = getattr(model.config, 'patch_order', 'raster')
    patch_order_perm = model.nape.embeddings.patch_embeddings.patch_order_perm.detach().cpu()
    print(f"Patch order: {patch_order}")

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    entries = manifest['data']
    sampled = random.sample(entries, min(args.num_clips, len(entries)))
    print(f"Sampled {len(sampled)} clips from {args.manifest}")

    cos_grids: List[np.ndarray] = []
    print()
    for i, entry in enumerate(sampled):
        wav_path = entry['wav']
        name = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            spec = extract_log_mel(
                wav_path,
                sample_rate=args.sample_rate, num_mel_bins=args.num_mel_bins,
                target_length=args.target_length, norm_mean=args.norm_mean,
                norm_std=args.norm_std, audio_duration=args.audio_duration,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(sampled)}] {name} — extraction error, skipping: {e}")
            continue

        z_hat, z = forward_clip(model, spec, device)
        expected = T_F * T_T
        if z_hat.shape[0] > expected:
            z_hat = z_hat[:expected]
            z     = z    [:expected]

        cos_grid = compute_prediction_quality_grid(z_hat, z, T_F, T_T, patch_order_perm)
        cos_grids.append(cos_grid)

        if i < args.per_clip_limit:
            out_path = os.path.join(args.output_dir, f"heatmap_clip{i:02d}_{name}.png")
            plot_prediction_heatmap(
                spec, cos_grid, T_F=T_F, T_T=T_T,
                patch_size=args.patch_size,
                output_path=out_path,
                title=f"Audio ID: {name}",
            )
            print(f"  [{i+1}/{len(sampled)}] {name} -> {os.path.basename(out_path)}")
        else:
            if i == args.per_clip_limit:
                print("  ... continuing without saving per-clip figures (raise --per_clip_limit to save more)")
            print(f"  [{i+1}/{len(sampled)}] {name}")

    if not cos_grids:
        raise RuntimeError("No clips were successfully processed.")

    # Averaged figure
    if len(cos_grids) > 1:
        cos_grids_stack = np.stack(cos_grids, axis=0)
        avg_path = os.path.join(args.output_dir, f"heatmap_averaged_n{len(cos_grids)}.png")
        plot_averaged_heatmap(
            cos_grids_stack, T_F=T_F, T_T=T_T,
            patch_size=args.patch_size,
            output_path=avg_path,
        )
        print(f"\nAveraged heatmap -> {avg_path}")

        mean_cos = float(np.nanmean(cos_grids_stack))
        per_patch_mean = np.nanmean(cos_grids_stack, axis=0)
        print(f"Overall mean cos sim: {mean_cos:.4f}")
        print(f"Per-patch mean range: [{float(np.nanmin(per_patch_mean)):.4f}, "
              f"{float(np.nanmax(per_patch_mean)):.4f}]")

    print("\nDone.")


if __name__ == '__main__':
    main()
