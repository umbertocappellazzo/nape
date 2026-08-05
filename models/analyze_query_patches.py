#!/usr/bin/env python
"""
analyze_query_patches.py
========================

Per-query-patch analysis of what Nape learns during pretraining.

For each (clip, query_patch) pair, produces a 3-panel figure:

  (i)   Spectrogram with the query patch highlighted by a box.
  (ii)  Attention map: which other patches the model attended to when
        computing the prediction at the query position. Averaged across
        attention heads and (optionally) across multiple layers.
  (iii) Embedding similarity map: cosine similarity between the predicted
        next-patch embedding (the prediction made at the query position)
        and every other patch embedding in the same clip.


Usage:
    python analyze_query_patches.py \\
        --checkpoint /abs/path/to/checkpoint \\
        --manifest   /abs/path/to/manifest.json \\
        --output_dir query_outputs \\
        --num_clips        5 \\
        --queries_per_clip 4 \\
        --query_positions  "2,15 2,45 5,15 5,45" \\
        --attention_pool   mean
"""

import argparse
import importlib
import json
import os
import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# Import the model (see analyze_pretrain.py for the rationale)
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
        waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
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
# Forward pass with attention extraction
# ----------------------------------------------------------------------------

def forward_clip_with_attention(
    model: NapeForPreTraining,
    spectrogram: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Returns:
        z:          [1+N, D]   input patch embeddings (CLS at position 0).
        h_pred:     [1+N, D]   encoder outputs after the prediction head g.
                               h_pred[k] is the predicted embedding for the
                               NEXT position (i.e. it predicts z[k+1]).
        attentions: list of [num_heads, 1+N, 1+N] attention tensors, one per
                    transformer layer.
    """
    model.eval()
    x = spectrogram.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.nape(x, output_attentions=True)
        z_input = outputs.input_embedding
        h_output = outputs.last_hidden_state
        attentions = outputs.attentions  # tuple of [1, H, 1+N, 1+N], len = num_layers

        if getattr(model, 'use_prediction_head', False) and hasattr(model, 'prediction_head'):
            h_pred = model.prediction_head(h_output)
        else:
            h_pred = h_output

    z          = z_input [0].float().cpu()
    h_pred     = h_pred  [0].float().cpu()
    attentions = [a[0].float().cpu() for a in attentions]
    return z, h_pred, attentions


# ----------------------------------------------------------------------------
# Helpers: patch grid <-> sequence position
# ----------------------------------------------------------------------------

def raster_idx_to_freq_time(raster_idx: int, T_T: int) -> Tuple[int, int]:
    return raster_idx // T_T, raster_idx % T_T

def freq_time_to_raster_idx(freq_idx: int, time_idx: int, T_T: int) -> int:
    return freq_idx * T_T + time_idx


def aggregate_attention(
    attentions: List[torch.Tensor],
    attention_pool: str,
    layer_indices: List[int] = None,
) -> torch.Tensor:
    """
    Aggregate per-layer attention tensors into a single [1+N, 1+N] map.

    Args:
        attentions: list of [num_heads, 1+N, 1+N] tensors, one per layer.
        attention_pool: how to combine layers — 'mean' (over all), 'first', 'first_4',
                        'middle', 'middle_4', 'last', or 'last_4'.
        layer_indices: explicit list of layer indices to include; overrides
                       attention_pool's choice when provided.
    Returns:
        A single [1+N, 1+N] attention map (heads always averaged).
    """
    L = len(attentions)
    mid = L // 2  # for 24 layers: mid=12
    if layer_indices is None:
        if attention_pool == 'first':
            layer_indices = [0]
        elif attention_pool == 'first_4':
            layer_indices = list(range(0, min(4, L)))
        elif attention_pool == 'middle':
            layer_indices = [mid]
        elif attention_pool == 'middle_4':
            # 4 layers centered on the midpoint. For 24 layers: [10,11,12,13].
            start = max(0, mid - 2)
            end = min(L, start + 4)
            layer_indices = list(range(start, end))
        elif attention_pool == 'last':
            layer_indices = [L - 1]
        elif attention_pool == 'last_4':
            layer_indices = list(range(max(0, L - 4), L))
        elif attention_pool == 'mean':
            layer_indices = list(range(L))
        else:
            raise ValueError(f"Unknown attention_pool: {attention_pool}")

    # Stack and average over heads then over selected layers
    stack = torch.stack([attentions[i].mean(dim=0) for i in layer_indices], dim=0)
    return stack.mean(dim=0)  # [1+N, 1+N]


# ----------------------------------------------------------------------------
# Per-layer analysis: grid of per-layer attention maps + summary statistics
# ----------------------------------------------------------------------------

def _compute_per_layer_stats(
    attentions: List[torch.Tensor],
    seq_pos: int,
    seq_patch_idx: int,
    perm_np: np.ndarray,
    inv_perm_np: np.ndarray,
    T_F: int,
    T_T: int,
):
    """
    For each layer, return (attn_raster, entropy_bits, mean_dist) where:
      attn_raster   — [N] array of attention from query to every patch, in raster
                      order (i.e. row-major (freq, time)), with future positions
                      zeroed out.
      entropy_bits  — Shannon entropy (in bits) of the renormalized attention
                      over past, non-self positions. Lower = more focused.
      mean_dist     — mean Euclidean distance in patch units between the query
                      and the attended patches (renormalized past, non-self).
    """
    # Query's (freq, time) in raster coords
    query_raster_idx = int(perm_np[seq_patch_idx])
    query_f, query_t = query_raster_idx // T_T, query_raster_idx % T_T

    # Pre-compute Euclidean distance from query to every raster patch
    fs = np.arange(T_F * T_T) // T_T
    ts = np.arange(T_F * T_T) % T_T
    dist = np.sqrt((fs - query_f) ** 2 + (ts - query_t) ** 2)

    # Masks (in raster order)
    future_mask = inv_perm_np > seq_patch_idx
    self_mask   = inv_perm_np == seq_patch_idx
    past_nonself_mask = (~future_mask) & (~self_mask)

    attn_per_layer, entropies, mean_dists = [], [], []
    for layer_attn in attentions:
        a = layer_attn.mean(dim=0)[seq_pos].numpy()  # [1+N] (heads averaged)
        a = a[1:]  # exclude CLS column → [N] in sequence order
        a_raster = np.empty_like(a)
        a_raster[perm_np] = a            # un-permute to raster order
        a_raster[future_mask] = 0.0      # defensive: zero out future positions
        attn_per_layer.append(a_raster)

        # Stats: renormalize past-non-self to a probability distribution
        a_past = a_raster.copy()
        a_past[~past_nonself_mask] = 0.0
        total = a_past.sum()
        if total > 1e-12:
            p = a_past / total
            with np.errstate(divide='ignore', invalid='ignore'):
                ent = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
            entropies.append(float(ent))
            mean_dists.append(float((p * dist).sum()))
        else:
            entropies.append(0.0)
            mean_dists.append(0.0)

    return attn_per_layer, entropies, mean_dists


def plot_per_layer_analysis(
    spectrogram: torch.Tensor,
    attentions: List[torch.Tensor],
    query_freq: int,
    query_time: int,
    T_F: int,
    T_T: int,
    output_path: str,
    patch_order_perm: torch.Tensor = None,
    title: str = None,
):
    """
    One figure summarising attention behaviour layer-by-layer for a single
    (clip, query) pair:
      - Top: grid of per-layer attention heatmaps (query → patches),
             one panel per layer.
      - Bottom: two line plots showing attention entropy and mean attention
             distance as a function of layer index.
    """
    raster_idx = freq_time_to_raster_idx(query_freq, query_time, T_T)
    if patch_order_perm is not None:
        perm = patch_order_perm if torch.is_tensor(patch_order_perm) \
            else torch.tensor(patch_order_perm, dtype=torch.long)
    else:
        perm = torch.arange(T_F * T_T)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel())
    seq_patch_idx = int(inv_perm[raster_idx].item())
    seq_pos = seq_patch_idx + 1  # +1 for CLS

    perm_np = perm.numpy()
    inv_perm_np = inv_perm.numpy()

    attn_per_layer, entropies, mean_dists = _compute_per_layer_stats(
        attentions, seq_pos, seq_patch_idx, perm_np, inv_perm_np, T_F, T_T,
    )
    L = len(attentions)

    # Grid layout: prefer 6 cols for typical layer counts (12, 24); fall back
    # to a near-square grid otherwise.
    if L % 6 == 0:
        grid_cols, grid_rows = 6, L // 6
    elif L % 4 == 0:
        grid_cols, grid_rows = 4, L // 4
    else:
        grid_cols = int(np.ceil(np.sqrt(L)))
        grid_rows = int(np.ceil(L / grid_cols))

    fig = plt.figure(figsize=(grid_cols * 2.2, grid_rows * 1.7 + 5))
    gs = fig.add_gridspec(
        grid_rows + 2, grid_cols,
        hspace=0.6, wspace=0.25,
        height_ratios=[1.0] * grid_rows + [1.1, 1.1],
    )

    # Per-layer attention heatmaps
    past_nonself_mask = (inv_perm_np <= seq_patch_idx) & (inv_perm_np != seq_patch_idx)
    for i in range(L):
        r, c = i // grid_cols, i % grid_cols
        ax = fig.add_subplot(gs[r, c])
        a_grid = attn_per_layer[i].reshape(T_F, T_T)

        # vmax = 99th percentile of past non-self attention (consistent with the
        # main plot in plot_query_analysis). Excluding self prevents the
        # query's own diagonal peak from dominating the colour scale.
        past_vals = attn_per_layer[i][past_nonself_mask]
        vmax = float(np.percentile(past_vals, 99)) if past_vals.size and past_vals.max() > 0 else 1.0
        if vmax <= 0:
            vmax = 1.0

        ax.imshow(a_grid, cmap='turbo', vmin=0.0, vmax=vmax,
                  aspect='auto', origin='lower')
        # Highlight the query patch
        ax.add_patch(Rectangle(
            (query_time - 0.5, query_freq - 0.5), 1, 1,
            edgecolor='red', facecolor='none', linewidth=1.5,
        ))
        ax.set_title(f'Layer {i}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Entropy line plot
    ax_ent = fig.add_subplot(gs[grid_rows, :])
    ax_ent.plot(range(L), entropies, 'o-', color='steelblue', linewidth=1.2)
    ax_ent.set_xlabel('Layer index', fontsize=9)
    ax_ent.set_ylabel('Attention entropy (bits)', fontsize=9)
    ax_ent.set_title('Attention entropy per layer — lower = more focused',
                     fontsize=10)
    ax_ent.set_xticks(range(L))
    ax_ent.tick_params(axis='both', labelsize=8)
    ax_ent.grid(True, alpha=0.3)

    # Mean attention distance line plot
    ax_dist = fig.add_subplot(gs[grid_rows + 1, :])
    ax_dist.plot(range(L), mean_dists, 'o-', color='firebrick', linewidth=1.2)
    ax_dist.set_xlabel('Layer index', fontsize=9)
    ax_dist.set_ylabel('Mean attention distance (patches)', fontsize=9)
    ax_dist.set_title('Mean Euclidean distance from query to attended patches '
                      'per layer', fontsize=10)
    ax_dist.set_xticks(range(L))
    ax_dist.tick_params(axis='both', labelsize=8)
    ax_dist.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=11, y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot: 3-panel figure for one query patch
# ----------------------------------------------------------------------------

def plot_query_analysis(
    spectrogram: torch.Tensor,
    z: torch.Tensor,           # [1+N, D] (includes CLS at 0)
    h_pred: torch.Tensor,      # [1+N, D]
    attentions_pooled: torch.Tensor,   # [1+N, 1+N] already pooled across layers/heads
    query_freq: int,
    query_time: int,
    T_F: int,
    T_T: int,
    patch_size: int,
    output_path: str,
    patch_order_perm: torch.Tensor = None,
    title: str = None,
    attention_power: float = 1.0,
):
    """
    Render the three-panel figure for a single (clip, query) example.
    """
    # Map (freq, time) -> raster patch index -> sequence position (account for CLS)
    raster_idx = freq_time_to_raster_idx(query_freq, query_time, T_T)

    if patch_order_perm is not None:
        perm = patch_order_perm if torch.is_tensor(patch_order_perm) else torch.tensor(patch_order_perm, dtype=torch.long)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel())
        seq_patch_idx = int(inv_perm[raster_idx].item())
    else:
        seq_patch_idx = raster_idx

    seq_pos = seq_patch_idx + 1  # +1 for CLS at position 0

    N = z.shape[0] - 1  # number of patches

    # ----- Embedding similarity map: cos sim(prediction_at_query, every patch z) -----
    pred_at_query = h_pred[seq_pos]                # [D] — predicts the next patch
    z_patches     = z[1:]                          # [N, D] — exclude CLS
    pred_norm = pred_at_query / (pred_at_query.norm() + 1e-8)
    z_norm    = z_patches    / (z_patches.norm(dim=-1, keepdim=True) + 1e-8)
    sim       = (z_norm @ pred_norm).numpy()       # [N]

    # ----- Attention map: from query position to each other patch -----
    attn_from_query = attentions_pooled[seq_pos].numpy()  # [1+N]
    attn_to_patches = attn_from_query[1:]                  # [N] — exclude CLS column

    # Un-permute both maps from sequence order back to raster order so the
    # 2D grid axes correspond to (freq_idx, time_idx).
    if patch_order_perm is not None:
        perm_np = perm.numpy()
        inv_perm_np = np.empty_like(perm_np)
        inv_perm_np[perm_np] = np.arange(perm_np.size)

        sim_raster      = np.empty_like(sim)
        attn_raster     = np.empty_like(attn_to_patches)
        sim_raster [perm_np] = sim
        attn_raster[perm_np] = attn_to_patches
        sim, attn_to_patches = sim_raster, attn_raster
    else:
        inv_perm_np = np.arange(N)

    # Future-position mask (causal mask). For each raster patch, its sequence
    # position is inv_perm_np[raster_idx]. A raster position is "future" if
    # its sequence position is strictly greater than the query's sequence
    # patch index (so the model never attended to it under causal masking).
    future_mask = inv_perm_np > seq_patch_idx   # [N], True for masked-out future
    # The query's own position (self-attention along the diagonal).
    self_mask   = inv_perm_np == seq_patch_idx
    # Defensive: explicitly zero the attention at future positions (they should
    # already be ~0 from causal softmax, but this guards against numerical
    # drift after averaging across heads/layers).
    attn_to_patches = attn_to_patches.copy()
    attn_to_patches[future_mask] = 0.0

    # Optional power transform on the attention values for visual contrast.
    # power = 1.0 is identity; smaller powers expand the low-value range.
    if attention_power != 1.0 and attention_power > 0:
        attn_to_patches = np.sign(attn_to_patches) * np.power(
            np.abs(attn_to_patches), attention_power
        )

    sim_grid  = sim          .reshape(T_F, T_T)
    attn_grid = attn_to_patches.reshape(T_F, T_T)

    # Scale the attention colour range using the past (non-future, non-self)
    # values only. Excluding self-attention is essential: the query token's
    # attention to itself usually dominates the diagonal and compresses every
    # other position into a narrow low band. We use a high percentile rather
    # than the absolute max to be robust against any remaining outliers.
    past_mask = (~future_mask) & (~self_mask)
    past_values = attn_to_patches[past_mask]
    if past_values.size > 0 and np.max(past_values) > 0:
        attn_vmax = float(np.percentile(past_values, 99))
        if attn_vmax <= 0:
            attn_vmax = float(np.max(past_values))
    else:
        # Edge case: query at the very first patch — no past context.
        attn_vmax = float(np.max(attn_to_patches)) if attn_to_patches.size > 0 else 1.0
    if attn_vmax <= 0:
        attn_vmax = 1.0

    # ----- Plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))

    spec_np = spectrogram.numpy()
    extent = [0, T_T * patch_size, 0, T_F * patch_size]

    def add_query_box(ax, edge_color='red', lw=2.0):
        rect = Rectangle(
            (query_time * patch_size, query_freq * patch_size),
            patch_size, patch_size,
            linewidth=lw, edgecolor=edge_color, facecolor='none',
        )
        ax.add_patch(rect)

    # Panel (i): spectrogram + query box
    ax = axes[0]
    ax.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    add_query_box(ax, edge_color='red', lw=2.5)
    ax.set_title('Query patch', fontsize = 15)
    ax.set_xlabel('Time frame', fontsize = 14)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    #ax.set_ylabel('Mel bin')

    # Panel (ii): attention map (future positions clamped to vmin = minimum colour)
    ax = axes[1]
    im = ax.imshow(attn_grid, aspect='auto', origin='lower', cmap='turbo',
                   interpolation='none', extent=extent,
                   vmin=0.0, vmax=attn_vmax)
    add_query_box(ax, edge_color='white', lw=2.0)
    ax.set_title('Attention map', fontsize = 15)
    ax.set_xlabel('Time frame', fontsize = 14)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    #ax.set_ylabel('Mel bin (patch grid)')
    plt.colorbar(im, ax=ax)

    # Panel (iii): embedding similarity map (computed against all patches, past and future)
    ax = axes[2]
    sim_vmin = float(np.nanmin(sim_grid))
    sim_vmax = float(np.nanmax(sim_grid))
    im = ax.imshow(sim_grid, aspect='auto', origin='lower', cmap='turbo',
                   interpolation='none', extent=extent,
                   vmin=sim_vmin, vmax=sim_vmax)
    add_query_box(ax, edge_color='white', lw=2.0)
    ax.set_title('Embedding-similarity map', fontsize = 15)
    ax.set_xlabel('Time frame', fontsize = 14)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    #ax.set_ylabel('Mel bin (patch grid)')
    plt.colorbar(im, ax=ax)

    if title:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Query position generation
# ----------------------------------------------------------------------------

def parse_query_positions(arg: str) -> List[Tuple[int, int]]:
    """Parse a string like '2,15 2,45 5,15 5,45' into a list of (freq, time) tuples."""
    positions = []
    for token in arg.strip().split():
        f_str, t_str = token.split(',')
        positions.append((int(f_str), int(t_str)))
    return positions


def auto_query_positions(K: int, T_F: int, T_T: int) -> List[Tuple[int, int]]:
    """Generate K reasonable default query positions on a fixed grid, avoiding edges."""
    # Avoid the first column (degenerate predictions from CLS only) and the
    # last column (padding). Stay one row in from the top and bottom too.
    pad_f, pad_t = 1, 3
    f_lo, f_hi = pad_f, T_F - 1 - pad_f
    t_lo, t_hi = pad_t, T_T - 1 - pad_t

    if K == 1:
        return [((f_lo + f_hi) // 2, (t_lo + t_hi) // 2)]
    if K == 2:
        return [(f_lo + 1, t_lo + 5), (f_hi - 1, t_hi - 5)]
    if K == 4:
        return [
            (f_lo + 1, t_lo + 5), (f_lo + 1, t_hi - 5),
            (f_hi - 1, t_lo + 5), (f_hi - 1, t_hi - 5),
        ]
    # General grid: roughly sqrt(K) rows x sqrt(K) cols
    rows = int(np.ceil(np.sqrt(K)))
    cols = int(np.ceil(K / rows))
    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= K:
                break
            f = f_lo + (r + 1) * (f_hi - f_lo) // (rows + 1)
            t = t_lo + (c + 1) * (t_hi - t_lo) // (cols + 1)
            positions.append((f, t))
    return positions[:K]


def energy_based_query_positions(
    spectrogram: torch.Tensor,
    K: int,
    T_F: int,
    T_T: int,
    patch_size: int = 16,
    freq_pad: int = 0,
    time_pad: int = 2,
    min_distance: int = 3,
) -> List[Tuple[int, int]]:
    """
    Pick K query patches with the highest mean log-mel energy. Encourages
    queries on actual acoustic events rather than on silence or ambient texture.

    Uses non-maximum suppression to enforce a minimum spatial separation
    between selected patches, so the K queries spread across distinct events
    rather than clustering on the loudest single event.

    Args:
        spectrogram:  [F, T] normalized log-mel spectrogram for one clip.
        K:            number of queries to return.
        T_F, T_T:     patch-grid dimensions.
        patch_size:   side length of a square patch in spectrogram units.
        freq_pad:     skip this many freq rows from each edge of the grid.
        time_pad:     skip this many time columns from each edge (defaults to 2
                      to avoid the causally-degenerate first column and the
                      zero-padded last column).
        min_distance: enforce this Chebyshev distance between selected patches.
    """
    spec_np = spectrogram.numpy()  # [F, T]
    F_total, T_total = spec_np.shape
    if F_total != T_F * patch_size or T_total != T_T * patch_size:
        raise ValueError(
            f"Spectrogram shape ({F_total}, {T_total}) is inconsistent with "
            f"patch grid ({T_F}, {T_T}) and patch size {patch_size}."
        )

    # Per-patch mean: reshape to (T_F, patch_size, T_T, patch_size) then mean
    # over the two within-patch dimensions.
    spec_4d = spec_np.reshape(T_F, patch_size, T_T, patch_size)
    patch_energy = spec_4d.mean(axis=(1, 3))  # [T_F, T_T]

    # Restrict to non-edge positions
    valid = np.zeros((T_F, T_T), dtype=bool)
    f_lo, f_hi = freq_pad, T_F - freq_pad
    t_lo, t_hi = time_pad, T_T - time_pad
    valid[f_lo:f_hi, t_lo:t_hi] = True

    candidates = np.where(valid, patch_energy, -np.inf)

    selected: List[Tuple[int, int]] = []
    for _ in range(K):
        flat_idx = int(np.argmax(candidates))
        f, t = np.unravel_index(flat_idx, candidates.shape)
        if not np.isfinite(candidates[f, t]):
            break
        selected.append((int(f), int(t)))
        # NMS: suppress everything within Chebyshev radius `min_distance`
        for df in range(-min_distance, min_distance + 1):
            for dt in range(-min_distance, min_distance + 1):
                ff, tt = f + df, t + dt
                if 0 <= ff < T_F and 0 <= tt < T_T:
                    candidates[ff, tt] = -np.inf

    return selected


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--manifest',    required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--num_clips',   type=int, default=5)
    parser.add_argument('--queries_per_clip', type=int, default=4,
                        help='Number of query patches per clip (ignored if '
                             '--query_positions is given).')
    parser.add_argument('--query_positions', type=str, default=None,
                        help='Manual query positions as space-separated "freq,time" pairs, '
                             'e.g. "2,15 2,45 5,15 5,45". Applies to every clip. Overrides '
                             '--query_strategy when provided.')
    parser.add_argument('--query_strategy', choices=['grid', 'energy'], default='grid',
                        help='How to choose query positions when --query_positions is not given. '
                             '"grid" uses a fixed set of positions on every clip. '
                             '"energy" picks the top-K highest-energy patches per clip via '
                             'NMS, so queries land on actual acoustic events.')
    parser.add_argument('--attention_pool',
                        choices=['mean', 'first', 'first_4', 'middle', 'middle_4', 'last', 'last_4'],
                        default='last',
                        help='How to aggregate attention across layers (heads always averaged). '
                             'Single layers: "first" (layer 0), "middle" (L//2), "last" (L-1). '
                             'Groups of 4: "first_4" / "middle_4" / "last_4". '
                             '"mean" averages all layers. Default "last" tends to show clearer '
                             'structure than "mean" because per-layer specialization is preserved.')
    parser.add_argument('--attention_power', type=float, default=1.0,
                        help='Power transform applied to attention values before plotting. '
                             'Use < 1.0 (e.g. 0.5 for sqrt) to expand the low-value range '
                             'and increase contrast.')
    parser.add_argument('--layer_analysis', action='store_true',
                        help='Also produce a per-layer analysis figure for each (clip, query) '
                             'pair: a grid of per-layer attention heatmaps plus summary curves '
                             '(entropy and mean attention distance vs. layer).')

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

    # Resolve manual query positions once (if given); energy/grid strategies
    # are resolved per-clip inside the loop.
    manual_positions = None
    if args.query_positions:
        manual_positions = parse_query_positions(args.query_positions)
        for (f, t) in manual_positions:
            if not (0 <= f < T_F and 0 <= t < T_T):
                raise ValueError(f"Query position ({f},{t}) out of bounds for grid ({T_F},{T_T})")
            if f == T_F - 1 and t == T_T - 1:
                print(f"  Warning: query at ({f},{t}) is the last raster position; "
                      f"its 'next-patch prediction' is degenerate.")
        print(f"Using manual query positions for every clip: {manual_positions}")
    else:
        print(f"Query strategy: {args.query_strategy} "
              f"(K = {args.queries_per_clip} per clip)")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint} (device={device})...")
    model = NapeForPreTraining.from_pretrained(args.checkpoint).to(device)
    model.eval()

    patch_order = getattr(model.config, 'patch_order', 'raster')
    patch_order_perm = model.nape.embeddings.patch_embeddings.patch_order_perm.detach().cpu()
    print(f"Patch order: {patch_order}")

    # Manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    entries = manifest['data']
    sampled = random.sample(entries, min(args.num_clips, len(entries)))
    print(f"Sampled {len(sampled)} clips from {args.manifest}\n")

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

        # Per-clip query positions
        if manual_positions is not None:
            query_positions = manual_positions
        elif args.query_strategy == 'grid':
            query_positions = auto_query_positions(args.queries_per_clip, T_F, T_T)
        elif args.query_strategy == 'energy':
            query_positions = energy_based_query_positions(
                spec, args.queries_per_clip,
                T_F=T_F, T_T=T_T, patch_size=args.patch_size,
            )
        else:
            raise ValueError(f"Unknown query strategy: {args.query_strategy}")

        z, h_pred, attentions = forward_clip_with_attention(model, spec, device)
        attn_pooled = aggregate_attention(attentions, args.attention_pool)

        print(f"[{i+1}/{len(sampled)}] {name}  —  queries: {query_positions}")
        for j, (qf, qt) in enumerate(query_positions):
            out_name = f"query_clip{i:02d}_q{j:02d}_f{qf:02d}_t{qt:02d}_{name}.png"
            out_path = os.path.join(args.output_dir, out_name)
            plot_query_analysis(
                spec, z, h_pred, attn_pooled,
                query_freq=qf, query_time=qt,
                T_F=T_F, T_T=T_T,
                patch_size=args.patch_size,
                output_path=out_path,
                patch_order_perm=patch_order_perm,
                attention_power=args.attention_power,
            )
            print(f"    -> {out_name}")

            if args.layer_analysis:
                layer_name = f"layer_clip{i:02d}_q{j:02d}_f{qf:02d}_t{qt:02d}_{name}.png"
                layer_path = os.path.join(args.output_dir, layer_name)
                plot_per_layer_analysis(
                    spec, attentions,
                    query_freq=qf, query_time=qt,
                    T_F=T_F, T_T=T_T,
                    output_path=layer_path,
                    patch_order_perm=patch_order_perm
                )
                print(f"    -> {layer_name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
