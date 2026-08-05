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
"""PyTorch Nape model."""

import contextlib
import math
from dataclasses import dataclass
from typing import  Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_nape import NapeConfig

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Norm factory: choose between LayerNorm and RMSNorm via config.norm_type
# ---------------------------------------------------------------------------

class _RMSNormFallback(nn.Module):
    """RMSNorm implementation for PyTorch < 2.4 (which lacks nn.RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).clamp_min(self.eps).sqrt()
        return (self.weight * (x.float() / rms)).to(x.dtype)


def _make_norm(hidden_size: int, config: NapeConfig,
               elementwise_affine: bool = True, bias: bool = True) -> nn.Module:
    """Build a normalization module based on `config.norm_type` ('layernorm' or 'rmsnorm').

    The optional `elementwise_affine` and `bias` flags pass through to LayerNorm;
    for RMSNorm, `bias` is ignored (RMSNorm has no bias by definition) and
    `elementwise_affine` controls whether the learnable scale is included.
    These flags exist for QK-Norm, which historically used
    `qk_norm_affine` / `qk_norm_bias` to control its sub-norms.
    """
    norm_type = getattr(config, 'norm_type', 'layernorm').lower()
    eps = config.layer_norm_eps
    if norm_type == 'layernorm':
        return nn.LayerNorm(hidden_size, eps=eps,
                            elementwise_affine=elementwise_affine, bias=bias)
    elif norm_type == 'rmsnorm':
        if hasattr(nn, 'RMSNorm'):
            return nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        return _RMSNormFallback(hidden_size, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}. Must be 'layernorm' or 'rmsnorm'.")


def _build_conv_stem(config: NapeConfig) -> nn.Module:
    """
    Build a multi-layer convolutional stem for patch embedding (alternative to
    the single non-overlapping Conv2d). Inspired by 'Early Convolutions Help
    Transformers See Better' (Xiao et al.), this is a 4-layer 3x3 conv stack
    with stride 2 in each layer, giving a total spatial reduction of 16x —
    matching the default patch_size=16 so the output sequence length is the
    same as the single-Conv2d patchifier.

    Channels grow gradually so the receptive field builds up over depth:
    1 → D/16 → D/8 → D/4 → D.

    Recipe: Conv → BN → GELU per layer, no norm or activation after the final
    layer (the encoder's pre-norm will normalize it). BatchNorm is fine here
    even with bf16 + DDP — it's used commonly in vision conv stems.
    """
    D = config.hidden_size
    C = config.num_channels
    if D % 16 != 0:
        raise ValueError(
            f"hidden_size={D} must be divisible by 16 for the conv stem channel schedule."
        )
    c1, c2, c3 = D // 16, D // 8, D // 4
    return nn.Sequential(
        nn.Conv2d(C,  c1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c1),
        nn.GELU(),
        nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c2),
        nn.GELU(),
        nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c3),
        nn.GELU(),
        nn.Conv2d(c3, D,  kernel_size=3, stride=2, padding=1, bias=False),
    )


def _resolve_grid_shape(config, freq_bins=None, time_frames=None):
    """
    Resolve the (num_patches_freq, num_patches_time) token grid shape based on
    `config.patch_embed_type`. For 'conv2d' / 'convstem' the grid is the
    standard 2D ViT-style (freq_bins / patch_size, time_frames / patch_size).
    For 'speech_stem' the grid is 1D (1, time_frames / downsample), because
    the speech stem flattens freq into channels so each token represents a
    single downsampled time step.

    Args:
        freq_bins, time_frames: input dimensions. If None, falls back to
            config.freq_bins / config.target_time_frames.
    """
    if freq_bins is None:
        freq_bins = config.freq_bins
    if time_frames is None:
        time_frames = config.target_time_frames

    patch_embed_type = getattr(config, 'patch_embed_type', 'conv2d').lower()
    if patch_embed_type == 'speech_stem':
        downsample = getattr(config, 'speech_stem_downsample', 2)
        return 1, time_frames // downsample
    return freq_bins // config.patch_size, time_frames // config.patch_size


class _SpeechStemPatchEmbedding(nn.Module):
    """
    Speech-encoder-style patch embedding for spectrograms (Gemma-3 inspired).

    Instead of ViT-style 2D patching, this applies a small 2D conv stack on
    the spectrogram and then flattens the frequency dimension into the channel
    dim, producing a 1D time-indexed token sequence (no frequency split).
    Each output token represents ALL frequencies at one downsampled time step.

    Architecture (channel progression 1 → 128 → 32, matching Gemma-3 audio):
        Conv2d(1, 128, k=3, s=(1, downsample), p=1) → BN → GELU
        Conv2d(128, 32, k=3, s=(1, 1), p=1)         → BN → GELU
    The temporal stride goes on the *first* conv so the second conv operates
    at the reduced rate. Frequency is not strided — kept full to be folded
    into channels.

    Output shape from `forward` is [B, hidden_size, 1, T / downsample] so the
    downstream flatten-then-reorder pipeline in NapePatchEmbeddings works
    unchanged.
    """

    def __init__(self, config: NapeConfig):
        super().__init__()
        D = config.hidden_size
        C = config.num_channels
        F = config.freq_bins
        downsample = getattr(config, 'speech_stem_downsample', 2)
        if config.target_time_frames % downsample != 0:
            raise ValueError(
                f"target_time_frames ({config.target_time_frames}) must be "
                f"divisible by speech_stem_downsample ({downsample})."
            )
        self.downsample = downsample
        self.freq_bins = F
        self.conv = nn.Sequential(
            nn.Conv2d(C, 128, kernel_size=3, stride=(1, downsample), padding=1, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=(1, 1), padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
        )
        # After conv: [B, 32, F, T/downsample]
        # Flatten freq into channels per time step, then project to hidden_size.
        self.proj = nn.Linear(32 * F, D)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # [B, C, F, T]
        x = self.conv(spectrogram)                       # [B, 32, F, T/downsample]
        B, C_out, F_out, T_out = x.shape
        # Move time forward, flatten (C_out, F_out) into one feature dim per time step
        x = x.permute(0, 3, 1, 2).reshape(B, T_out, C_out * F_out)
        x = self.proj(x)                                  # [B, T_out, D]
        # Reshape to [B, D, 1, T_out] so downstream flatten + reorder pipeline
        # in NapePatchEmbeddings.forward works unchanged.
        return x.transpose(1, 2).unsqueeze(2)


class _CausalTransformerPredictor(nn.Module):
    """
    Transformer-based prediction head with causal self-attention, matching the
    style of predictors used in JEPA-family methods (I-JEPA, A-JEPA, V-JEPA,
    LeWorldModel). The causal mask on the predictor is essential: without it,
    the token at position t could attend to token t+1 and directly read the
    target it is supposed to predict, causing the loss to saturate and the
    representations to collapse — the same failure mode observed when the
    encoder's causal mask is removed (see Section 4.2 in the paper).

    Structure:
        Linear(D, predictor_dim)  # input projection (skipped if predictor_dim == D)
        [pre-norm Transformer block with causal self-attention] x num_layers
        Linear(predictor_dim, D)  # output projection (skipped if predictor_dim == D)

    Defaults (configurable via config):
        predictor_dim:        config.predictor_dim         (default: hidden_size)
        num_layers:           config.predictor_num_layers  (default: 2)
        num_heads:            config.predictor_num_heads   (default: config.num_attention_heads)
        FFN dim:              4 * predictor_dim
        activation:           GELU
        norm:                 pre-norm

    With the defaults above, the predictor operates in the same dimension as
    the encoder — no input/output projections are needed — and uses a
    Transformer stack shallower than the encoder itself.
    """

    def __init__(self, config: NapeConfig):
        super().__init__()
        D = config.hidden_size
        predictor_dim = getattr(config, 'predictor_dim', D)
        num_layers = getattr(config, 'predictor_num_layers', 2)
        num_heads = getattr(config, 'predictor_num_heads', config.num_attention_heads)

        self.input_proj = nn.Linear(D, predictor_dim) if predictor_dim != D else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=4 * predictor_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm, matching the encoder design
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(predictor_dim, D) if predictor_dim != D else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x = self.input_proj(x)
        seq_len = x.size(1)
        # Explicit causal mask. Positions strictly ahead are set to -inf; the
        # diagonal (self) is allowed. This prevents the predictor at position t
        # from attending to positions >= t+1, which would leak the target.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.output_proj(x)
        return x


def _build_prediction_head(config: NapeConfig) -> nn.Module:
    """
    Build the optional prediction MLP applied to the encoder output before the
    next-embedding prediction loss. Choice of architecture controlled by
    `config.prediction_head_type`.

    Variants:
        'mlp2' (default):
            Linear(D, D) -> GELU -> Linear(D, D)
            The original simple 2-layer head.

        'simsiam':
            Linear(D, D) -> Norm -> GELU -> Linear(D, D) -> Norm -> GELU -> Linear(D, D)
            3-layer with intermediate normalization. Final norm is intentionally
            omitted — SimSiam (Chen & He, 2020) reports that applying a norm to
            the predictor's output can cause training instability.

        'byol':
            Linear(D, 4D) -> Norm -> GELU -> Linear(4D, D)
            2-layer bottleneck/expansion structure used in BYOL (Grill et al.,
            2020). Hidden dim is 4x for added capacity; norm and activation in
            the middle, no final norm.

        'transformer':
            6-layer pre-norm Transformer with causal self-attention, in the style
            of JEPA-family predictors. Requires the causal mask to prevent target
            leakage. See `_CausalTransformerPredictor` for details and sizing.

    The norm uses `_make_norm`, so it tracks `config.norm_type` and stays
    consistent with the rest of the encoder (LayerNorm by default; RMSNorm if
    `config.norm_type='rmsnorm'`).
    """
    head_type = getattr(config, 'prediction_head_type', 'mlp2').lower()
    D = config.hidden_size

    if head_type == 'mlp2':
        return nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
    elif head_type == 'simsiam':
        return nn.Sequential(
            nn.Linear(D, D),
            _make_norm(D, config),
            nn.GELU(),
            nn.Linear(D, D),
            _make_norm(D, config),
            nn.GELU(),
            nn.Linear(D, D),
        )
    elif head_type == 'byol':
        return nn.Sequential(
            nn.Linear(D, 4 * D),
            _make_norm(4 * D, config),
            nn.GELU(),
            nn.Linear(4 * D, D),
        )
    elif head_type == 'transformer':
        return _CausalTransformerPredictor(config)
    else:
        raise ValueError(
            f"Unknown prediction_head_type: {head_type!r}. "
            f"Must be 'mlp2', 'simsiam', 'byol', or 'transformer'."
        )


# ---------------------------------------------------------------------------
# Positional embeddings (2D RoPE for freq × time grid)
# ---------------------------------------------------------------------------

def get_patch_order_permutation(
    num_patches_freq: int,
    num_patches_time: int,
    patch_order: str,
) -> list[int]:
    """
    Compute a permutation that maps raster (row-major) flat indices to the
    desired patch ordering.

    Returns:
        List of length N = num_patches_freq * num_patches_time, where
        result[i] is the raster-order flat index of the i-th patch in the
        desired ordering.
    """
    F, T = num_patches_freq, num_patches_time

    if patch_order == "raster":
        # Row-major: (f0,t0), (f0,t1), ..., (f0,tT-1), (f1,t0), ...
        return list(range(F * T))

    elif patch_order == "time_major":
        # Column-major: (t0,f0), (t0,f1), ..., (t0,fF-1), (t1,f0), ...
        indices = []
        for t in range(T):
            for f in range(F):
                indices.append(f * T + t)
        return indices

    elif patch_order == "zigzag":
        # Zigzag raster: row 0 left-to-right, row 1 right-to-left, etc.
        # Preserves spatial locality at row transitions.
        indices = []
        for f in range(F):
            time_range = range(T) if f % 2 == 0 else range(T - 1, -1, -1)
            for t in time_range:
                indices.append(f * T + t)
        return indices

    elif patch_order == "diagonal":
        # Anti-diagonal scan: sorted by (f + t), tiebreaker by f ascending.
        # Provides balanced context in both frequency and time dimensions.
        patches = []
        for f in range(F):
            for t in range(T):
                flat_idx = f * T + t
                patches.append((f + t, f, flat_idx))
        patches.sort(key=lambda x: (x[0], x[1]))
        return [p[2] for p in patches]

    else:
        raise ValueError(f"Unknown patch_order: {patch_order}")


def get_patches_center_coordinates(
    num_patches_freq: int,
    num_patches_time: int,
    patch_order: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute 2D centre coordinates for spectrogram patches, normalised to [-1, +1].

    Returns:
        Tensor of shape (num_patches_freq * num_patches_time, 2) where each row is (freq, time).
        The ordering follows `patch_order`.
    """
    coords_freq = torch.arange(0.5, num_patches_freq, dtype=dtype, device=device) / num_patches_freq
    coords_time = torch.arange(0.5, num_patches_time, dtype=dtype, device=device) / num_patches_time

    # (F, T, 2) grid in raster order
    coords = torch.stack(
        torch.meshgrid(coords_freq, coords_time, indexing="ij"), dim=-1
    )
    coords = coords.flatten(0, 1)  # (N, 2) in raster order

    # Reorder according to patch_order
    perm = get_patch_order_permutation(num_patches_freq, num_patches_time, patch_order)
    coords = coords[perm]

    coords = 2.0 * coords - 1.0   # shift [0, 1] → [-1, +1]
    return coords


class NapeRopePositionEmbedding(nn.Module):
    """2D Rotary Position Embedding for spectrogram patches."""

    inv_freq: torch.Tensor
    patch_coords_cached: torch.Tensor

    def __init__(self, config: NapeConfig):
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute and cache the patch center coordinates for the expected
        # spectrogram grid size. This avoids the Python-level recomputation
        # (including a sort/loop in get_patch_order_permutation for non-raster
        # orderings) on every forward pass. The .forward() will fall back to
        # the on-the-fly computation if a different grid size is ever seen.
        # _resolve_grid_shape handles the speech_stem case where the token grid
        # is 1D in time (1 × T/downsample) rather than 2D.
        num_patches_freq, num_patches_time = _resolve_grid_shape(config)
        coords = get_patches_center_coordinates(
            num_patches_freq, num_patches_time,
            config.patch_order,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        self.register_buffer("patch_coords_cached", coords, persistent=False)
        self._cached_grid_shape = (num_patches_freq, num_patches_time)

    def forward(self, spectrogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectrogram: [B, C, F, T] input spectrogram (used to infer grid size).
        Returns:
            (cos, sin) each of shape (num_patches, head_dim).
        """
        _, _, freq_bins, time_frames = spectrogram.shape
        num_patches_freq, num_patches_time = _resolve_grid_shape(
            self.config, freq_bins=freq_bins, time_frames=time_frames,
        )

        device = spectrogram.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # Fast path: grid size matches the one cached at init → just move
            # the cached tensor to the right device (which is a no-op once it
            # has been migrated by the first call after .to(device)).
            if (num_patches_freq, num_patches_time) == self._cached_grid_shape:
                patch_coords = self.patch_coords_cached
                if patch_coords.device != device:
                    patch_coords = patch_coords.to(device=device)
            else:
                # Slow fallback for variable-grid inference (rare).
                patch_coords = get_patches_center_coordinates(
                    num_patches_freq, num_patches_time,
                    self.config.patch_order,
                    dtype=torch.float32, device=device,
                )

            # (N, 2, head_dim/4) → (N, head_dim/2) → (N, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = spectrogram.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE only to patch tokens, skipping prefix tokens (CLS)."""
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix, q_patches), dim=-2)
    k = torch.cat((k_prefix, k_patches), dim=-2)
    return q, k


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaseModelOutputWithEmbedding(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    input_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class EmbeddedModelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    frame_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# ---------------------------------------------------------------------------
# Patch embeddings
# ---------------------------------------------------------------------------

class NapePatchEmbeddings(nn.Module):
    """
    Turn a spectrogram [B, 1, F, T] into patch embeddings [B, N, D].

    Three patch-embedding architectures are supported via config.patch_embed_type:
      - 'conv2d' (default): single non-overlapping Conv2d with kernel=stride=patch_size.
                            Produces a 2D token grid (freq × time) following the ViT
                            convention.
      - 'convstem': 4-layer 3x3 stride-2 conv stack (ConvViT-style). Also a 2D token
                    grid; total spatial reduction is 16x to match conv2d.
      - 'speech_stem': speech-encoder-style stack (Gemma-3 inspired). Outputs a 1D
                       time-indexed token sequence (no frequency split). Each token
                       represents all frequencies at one downsampled time step. Token
                       grid is (1, T // speech_stem_downsample).
    """

    def __init__(self, config: NapeConfig):
        super().__init__()
        self.config = config
        self.freq_bins = config.freq_bins
        self.target_time_frames = config.target_time_frames
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.patch_order = config.patch_order

        self.num_patches_freq, self.num_patches_time = _resolve_grid_shape(config)
        self.num_patches = self.num_patches_freq * self.num_patches_time

        patch_embed_type = getattr(config, 'patch_embed_type', 'conv2d').lower()
        self.patch_embed_type = patch_embed_type

        if patch_embed_type == 'conv2d':
            self.projection = nn.Conv2d(
                config.num_channels,
                config.hidden_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            )
        elif patch_embed_type == 'convstem':
            if config.patch_size != 16:
                raise ValueError(
                    f"patch_embed_type='convstem' assumes patch_size=16 (4 layers of "
                    f"stride-2 conv = 16x total). Got patch_size={config.patch_size}."
                )
            self.projection = _build_conv_stem(config)
        elif patch_embed_type == 'speech_stem':
            self.projection = _SpeechStemPatchEmbedding(config)
        else:
            raise ValueError(
                f"Unknown patch_embed_type: {patch_embed_type!r}. "
                f"Must be 'conv2d', 'convstem', or 'speech_stem'."
            )

        # Precompute patch order permutation (maps raster flat indices to desired order).
        # For speech_stem (1 × T_grid), this is the identity for all patch orders.
        perm = get_patch_order_permutation(
            self.num_patches_freq, self.num_patches_time, self.patch_order
        )
        self.register_buffer("patch_order_perm", torch.tensor(perm, dtype=torch.long), persistent=False)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: [B, C, F, T]
        Returns:
            embeddings: [B, N, D] where N = num_patches_freq * num_patches_time
        """
        batch_size, num_channels, freq_bins, time_frames = spectrogram.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels but got {num_channels}."
            )

        # [B, D, F', T'] where F' = freq_bins / patch_size, T' = time_frames / patch_size
        embeddings = self.projection(spectrogram)

        # Flatten spatial dims to raster order: [B, D, N]
        embeddings = embeddings.flatten(2)

        # Reorder according to patch_order
        embeddings = embeddings[:, :, self.patch_order_perm]

        # Transpose to sequence: [B, N, D]
        embeddings = embeddings.transpose(1, 2)
        return embeddings


class NapeEmbeddings(nn.Module):
    """
    Construct the CLS token, optional register tokens, and patch embeddings.
    Positional information is handled with RoPE inside attention.
    """

    def __init__(self, config: NapeConfig, use_mask_token: bool = False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = NapePatchEmbeddings(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        # Register tokens (Darcet et al., 2024): learnable tokens placed after CLS
        # and before patch tokens. They receive no positional encoding (no RoPE)
        # and act as global information aggregation slots.
        self.num_register_tokens = getattr(config, 'num_register_tokens', 0)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, self.num_register_tokens, config.hidden_size) * 0.02
            )

        # Optional learnable absolute positional embeddings (ablation against RoPE).
        # Covers the full sequence: [CLS, register_1, ..., register_R, patch_1, ..., patch_N].
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'rope').lower()
        if self.position_embedding_type == 'absolute':
            num_patches = self.patch_embeddings.num_patches
            total_seq_len = 1 + self.num_register_tokens + num_patches
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, total_seq_len, config.hidden_size)
            )
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        elif self.position_embedding_type != 'rope':
            raise ValueError(
                f"Unknown position_embedding_type: {self.position_embedding_type!r}. "
                f"Must be 'rope' or 'absolute'."
            )

    def forward(
        self,
        spectrogram: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spectrogram.shape[0]
        embeddings = self.patch_embeddings(spectrogram)
        embeddings_clean = embeddings

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings_clean = torch.cat((cls_tokens, embeddings_clean), dim=1)

        # Insert register tokens after CLS, before patches: [CLS, reg_1, ..., reg_R, patch_1, ...]
        # Registers get no RoPE (handled automatically by apply_rotary_pos_emb
        # which skips all prefix tokens before the patch tokens).
        if self.num_register_tokens > 0:
            reg_tokens = self.register_tokens.expand(batch_size, -1, -1)
            embeddings = torch.cat((embeddings[:, :1], reg_tokens, embeddings[:, 1:]), dim=1)
            embeddings_clean = torch.cat((embeddings_clean[:, :1], reg_tokens, embeddings_clean[:, 1:]), dim=1)

        # Add absolute positional embeddings if enabled (ablation against RoPE).
        if self.position_embedding_type == 'absolute':
            embeddings = embeddings + self.position_embeddings
            embeddings_clean = embeddings_clean + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings, embeddings_clean


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    output_attentions: bool = False,
    is_causal: bool = False,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if is_causal:
        q_len, k_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.full(
            (q_len, k_len), fill_value=float("-inf"), device=attn_weights.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    outputs = (attn_output, attn_weights) if output_attentions else (attn_output, None)
    return outputs


class NapeSelfAttention(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} is not a multiple of "
                f"the number of attention heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size ** -0.5
        self.is_causal = config.is_causal

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        if config.qk_norm:
            self.q_norm = _make_norm(
                self.attention_head_size, config,
                elementwise_affine=config.qk_norm_affine,
                bias=config.qk_norm_bias,
            )
            self.k_norm = _make_norm(
                self.attention_head_size, config,
                elementwise_affine=config.qk_norm_affine,
                bias=config.qk_norm_bias,
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = hidden_states.shape[0]
        new_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)

        # QK Norm
        query_layer = self.q_norm(query_layer)
        key_layer = self.k_norm(key_layer)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        context_layer, attention_probs = eager_attention_forward(
            self,
            query_layer, key_layer, value_layer,
            head_mask,
            output_attentions=output_attentions,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_shape)
        return context_layer, attention_probs


class NapeSelfOutput(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class NapeAttention(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        self.attention = NapeSelfAttention(config)
        self.output = NapeSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, ...]:
        self_output = self.attention(hidden_states, head_mask, output_attentions, position_embeddings)
        attn_output = self.output(self_output[0], hidden_states)
        return (attn_output,) + self_output[1:]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class NapeIntermediate(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        self.use_gated_mlp = config.use_gated_mlp
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        if self.use_gated_mlp:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_out = self.up_proj(hidden_states)
        if self.use_gated_mlp:
            gate = self.gate_proj(hidden_states)
            hidden_states = self.act_fn(gate) * up_out
        else:
            hidden_states = self.act_fn(up_out)
        return hidden_states


# ---------------------------------------------------------------------------
# LayerScale & DropPath
# ---------------------------------------------------------------------------

class NapeLayerScale(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        if config.layerscale_value is not None:
            self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))
        else:
            self.lambda1 = None

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.lambda1 is None:
            return hidden_state
        return hidden_state * self.lambda1


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output


class NapeDropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Transformer layer & encoder
# ---------------------------------------------------------------------------

class NapeOutput(nn.Module):
    def __init__(self, config: NapeConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_scale = NapeLayerScale(config)
        self.drop_path = NapeDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_scale(hidden_states)
        hidden_states = input_tensor + self.drop_path(hidden_states)
        return hidden_states


class NapeLayer(nn.Module):
    def __init__(self, config: NapeConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.attention = NapeAttention(config)
        self.layer_scale = NapeLayerScale(config)
        self.drop_path = NapeDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = NapeIntermediate(config)
        self.output = NapeOutput(config, drop_path_rate)
        self.layernorm_before = _make_norm(config.hidden_size, config)
        self.layernorm_after = _make_norm(config.hidden_size, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, ...]:
        # Pre-norm + self-attention
        hidden_states_norm = self.layernorm_before(hidden_states)
        self_attention_output = self.attention(
            hidden_states_norm, head_mask, output_attentions, position_embeddings,
        )
        attention_output = self_attention_output[0]
        attention_output = self.layer_scale(attention_output)
        extra_outputs = self_attention_output[1:]

        # First residual connection
        hidden_states = hidden_states + self.drop_path(attention_output)

        # Pre-norm + FFN
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # Second residual connection (inside NapeOutput)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,) + extra_outputs


class NapeEncoder(nn.Module):
    def __init__(self, config: NapeConfig):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_hidden_layers)]
        self.layer = nn.ModuleList([
            NapeLayer(config, drop_path_rate=dpr[i])
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> BaseModelOutput:
        all_attentions = () if output_attentions else None
        all_hidden_states = [] if self.config.output_hidden_states else None

        if self.config.output_hidden_states:
            all_hidden_states.append(hidden_states)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_output = layer_module(
                hidden_states, layer_head_mask, output_attentions, position_embeddings,
            )
            hidden_states = layer_output[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_output[1],)
            if self.config.output_hidden_states:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_attentions,
            hidden_states=all_hidden_states,
        )


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

class NapePreTrainedModel(PreTrainedModel):
    config_class = NapeConfig
    base_model_prefix = "nape"
    main_input_name = "spectrogram"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or module.__class__.__name__ in ('RMSNorm', '_RMSNormFallback'):
            # Handles nn.LayerNorm, nn.RMSNorm (torch >= 2.4), and our fallback.
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, NapeEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0, std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
            if module.mask_token is not None:
                module.mask_token.data.zero_()
        elif isinstance(module, NapeLayerScale):
            if module.lambda1 is not None:
                module.lambda1.data.fill_(self.config.layerscale_value)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NapeModel(NapePreTrainedModel):
    def __init__(self, config: NapeConfig, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = NapeEmbeddings(config, use_mask_token=use_mask_token)

        # Only build RoPE if it's actually going to be used. With absolute
        # positional embeddings, RoPE is not applied and the rope_embeddings
        # module is omitted entirely.
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'rope').lower()
        if self.position_embedding_type == 'rope':
            self.rope_embeddings = NapeRopePositionEmbedding(config)
        else:
            self.rope_embeddings = None

        self.encoder = NapeEncoder(config)
        self.layernorm = _make_norm(config.hidden_size, config)
        self.post_init()

    def get_input_embeddings(self) -> NapePatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithEmbedding:
        if spectrogram is None:
            raise ValueError("You have to specify spectrogram")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Get the projection's dtype to match input spectrogram dtype.
        # Works for any patch_embed_type — conv2d (a bare nn.Conv2d), convstem
        # (an nn.Sequential), or speech_stem (a custom module). Just grab the
        # first parameter's dtype.
        expected_dtype = next(self.embeddings.patch_embeddings.projection.parameters()).dtype
        if spectrogram.dtype != expected_dtype:
            spectrogram = spectrogram.to(expected_dtype)

        embedding_input, embedding_clean = self.embeddings(spectrogram, bool_masked_pos=bool_masked_pos)

        # Position embeddings: RoPE (passed to attention layers) or None (when
        # absolute positional embeddings were already added inside self.embeddings).
        position_embeds = self.rope_embeddings(spectrogram) if self.rope_embeddings is not None else None

        encoder_outputs: BaseModelOutput = self.encoder(
            embedding_input,
            head_mask=head_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeds,
        )
        sequence_output = self.layernorm(encoder_outputs.last_hidden_state)

        return BaseModelOutputWithEmbedding(
            last_hidden_state=sequence_output,
            input_embedding=embedding_clean,
            attentions=encoder_outputs.attentions,
            hidden_states=encoder_outputs.hidden_states,
        )


# ---------------------------------------------------------------------------
# Pre-training head (next-embedding prediction)
# ---------------------------------------------------------------------------

class NapeForPreTraining(NapePreTrainedModel):
    """
    Pretraining wrapper around NapeModel.

    Loss types:
      - 'cosine': negative cosine similarity between predicted and target
                  next-step embeddings. Targets are normalized to the unit
                  sphere.
      - 'mse':    mean-squared error between predicted and target next-step
                  embeddings. Equivalent to LeWM/JEPA-style loss.

    Anti-collapse mechanisms (independent and stackable):
      - Stop-gradient on the target side of the prediction loss (default ON;
        the standard NEPA / I-JEPA recipe). Set
        `disable_target_stop_gradient=True` to remove it.

    The optional MLP `prediction_head` (cosine path only) is kept because
    SimSiam-style projector designs may help even without EMA teachers.
    """

    def __init__(self, config: NapeConfig):
        super().__init__(config)

        # Enable mask token if masking is used during pretraining
        self.mask_ratio = getattr(config, 'mask_ratio', 0.0)
        use_mask_token = self.mask_ratio > 0.0
        self.nape = NapeModel(config, use_mask_token=use_mask_token)

        self.loss_type = getattr(config, 'loss_type', 'cosine')
        if self.loss_type not in ('cosine', 'mse', 'l1', 'cross_entropy'):
            raise ValueError(
                f"Unknown loss_type: {self.loss_type!r}. "
                f"Must be 'cosine', 'mse', 'l1', or 'cross_entropy'."
            )

        # What to use as the prediction target.
        #   - 'patch_embedding' (default): output of the patch embedding layer (the
        #     patchifier), in the encoder's sequence space. Standard NAPE target.
        #   - 'raw_mel': raw mel-spectrogram patch values (patch_size^2 * num_channels
        #     per patch). Per-patch normalization is applied if config.raw_mel_normalize
        #     is True (MAE-style). Requires a separate Linear that projects the encoder
        #     output (after prediction_head) down to the raw-patch dimensionality.
        #   - 'encoder_layer': output of a specific encoder layer (specified by
        #     config.target_layer_index, 0-indexed where 0 = first transformer block).
        #     Predicts in deep-feature space rather than embedding/input space. WARNING:
        #     stop-gradient alone is insufficient anti-collapse for this target — the
        #     encoder produces both pred and target, so it can drift to make both
        #     trivial. Use this for an ablation that demonstrates the collapse risk,
        #     or layer on EMA / SIGReg for a non-collapsing variant.
        self.target_type = getattr(config, 'target_type', 'patch_embedding').lower()
        self.raw_mel_normalize = getattr(config, 'raw_mel_normalize', True)
        if self.target_type not in ('patch_embedding', 'raw_mel', 'encoder_layer'):
            raise ValueError(
                f"Unknown target_type: {self.target_type!r}. "
                f"Must be 'patch_embedding', 'raw_mel', or 'encoder_layer'."
            )
        if self.target_type == 'raw_mel':
            patch_embed_type = getattr(config, 'patch_embed_type', 'conv2d').lower()
            if patch_embed_type == 'speech_stem':
                raise ValueError(
                    "target_type='raw_mel' is incompatible with "
                    "patch_embed_type='speech_stem': the speech stem produces 1D "
                    "time tokens that span all frequencies, so there's no "
                    "well-defined 'raw mel patch' to predict per token."
                )
            patch_dim = config.patch_size ** 2 * config.num_channels
            # Project the predictor output (which lives in hidden_size dim) to
            # the raw-patch dimensionality so it can match the unfolded mel.
            self.target_projection = nn.Linear(config.hidden_size, patch_dim)
            if self.loss_type == 'cosine':
                logger.warning(
                    "target_type='raw_mel' with loss_type='cosine': cosine similarity "
                    "is unusual in raw mel-energy space (values aren't unit-normalized). "
                    "Consider 'mse' or 'l1'."
                )
        elif self.target_type == 'encoder_layer':
            # Default to the middle layer if not specified.
            target_layer_index = getattr(config, 'target_layer_index', None)
            if target_layer_index is None:
                target_layer_index = config.num_hidden_layers // 2
            if not (0 <= target_layer_index < config.num_hidden_layers):
                raise ValueError(
                    f"target_layer_index={target_layer_index} out of range. "
                    f"Must be in [0, {config.num_hidden_layers - 1}]."
                )
            self.target_layer_index = target_layer_index
            # We need the encoder to expose all intermediate hidden states so
            # we can pluck the chosen one out. Force this on the config.
            config.output_hidden_states = True
            logger.warning(
                f"target_type='encoder_layer' with target_layer_index={target_layer_index}: "
                f"stop-gradient alone may NOT prevent representational collapse. The "
                f"encoder produces both prediction and target, so it can drift toward "
                f"degenerate solutions. If training diverges or features collapse, you "
                f"likely need an EMA teacher (I-JEPA-style) or distributional "
                f"regularizer (SIGReg)."
            )


        self.use_prediction_head = getattr(config, 'use_prediction_head', False)
        if self.use_prediction_head:
            self.prediction_head = _build_prediction_head(config)

        # Whether to detach the target embedding before the prediction loss.
        # Default True is the standard NAPE / I-JEPA recipe (stop-gradient as
        # the anti-collapse mechanism). Set False for LeWM-style training,
        # which lets gradient flow through both sides of the prediction loss
        # and relies on SIGReg to prevent collapse.
        self.disable_target_stop_gradient = getattr(
            config, 'disable_target_stop_gradient', False
        )

        # Whether to apply the autoregressive shift in the prediction loss.
        # Default True: position t's encoder output predicts position t+1's
        # patch embedding (next-embedding prediction — the NAPE objective).
        # Set False for the no-shift ablation: position t predicts position t.
        # This reduces the objective to an identity mapping with no meaningful
        # prediction signal and should fail to converge.
        self.use_autoregressive_shift = getattr(config, 'use_autoregressive_shift', True)


        self.post_init()

    def _extract_raw_patches(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Unfold the spectrogram into raw non-overlapping patches and reorder them
        to match the encoder's sequence ordering (patch_order). Used as the
        target when target_type='raw_mel'.

        Returns:
            [B, N, patch_size**2 * num_channels] tensor.
        """
        patch_size = self.nape.embeddings.patch_embeddings.patch_size
        # spectrogram: [B, C, F, T] → [B, C, F/p, p, T/p, p]
        patches = spectrogram.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # → [B, F/p, T/p, C, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # → [B, F/p * T/p, C * p * p]
        patches = patches.flatten(1, 2).flatten(2)
        # Reorder raster → patch_order to match the encoder sequence
        perm = self.nape.embeddings.patch_embeddings.patch_order_perm
        return patches[:, perm, :]

    def forward(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> EmbeddedModelingOutput:

        # Optional patch masking during training
        bool_masked_pos = None
        if self.mask_ratio > 0.0 and self.training:
            batch_size = spectrogram.shape[0]
            num_patches = self.nape.embeddings.patch_embeddings.num_patches
            num_masked = int(num_patches * self.mask_ratio)

            noise = torch.rand(batch_size, num_patches, device=spectrogram.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            bool_masked_pos = torch.zeros(batch_size, num_patches,
                                          dtype=torch.bool, device=spectrogram.device)
            bool_masked_pos.scatter_(1, ids_shuffle[:, :num_masked], True)

        outputs: BaseModelOutputWithEmbedding = self.nape(
            spectrogram,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

        sequence_input = outputs.input_embedding   # patch-embedding outputs (default target)
        sequence_output = outputs.last_hidden_state  # backbone output (predictions)

        # Optional projector
        if self.use_prediction_head:
            sequence_output_proj = self.prediction_head(sequence_output)
        else:
            sequence_output_proj = sequence_output

        # Determine the prediction target based on target_type.
        # 'patch_embedding' (default): target = sequence_input (patch embedding outputs).
        # 'raw_mel': target = unfolded raw mel patches (optionally per-patch normalized),
        # and the prediction is projected from hidden_size down to the patch dim.
        # 'encoder_layer': target = output of a specific encoder layer (deep features).
        if self.target_type == 'raw_mel':
            target_full = self._extract_raw_patches(spectrogram)            # [B, N, P]
            if self.raw_mel_normalize:
                # MAE-style per-patch normalization
                mean = target_full.mean(dim=-1, keepdim=True)
                var = target_full.var(dim=-1, keepdim=True)
                target_full = (target_full - mean) / (var + 1e-6).sqrt()
            # Pad prefix to align with sequence_input layout (CLS + register tokens)
            num_prefix = sequence_input.shape[1] - target_full.shape[1]
            if num_prefix > 0:
                prefix = torch.zeros(
                    target_full.shape[0], num_prefix, target_full.shape[2],
                    device=target_full.device, dtype=target_full.dtype,
                )
                target_full = torch.cat((prefix, target_full), dim=1)
            # Project the predictor output to the raw patch dimensionality.
            sequence_output_proj = self.target_projection(sequence_output_proj)
        elif self.target_type == 'encoder_layer':
            # outputs.hidden_states is a tuple:
            #   [0] = input to first transformer block (= patch embedding + CLS + pos),
            #   [k] = output of (k-1)-th transformer block for k = 1..L.
            # So target_layer_index=k (0-indexed transformer block) -> hidden_states[k+1].
            if outputs.hidden_states is None:
                raise RuntimeError(
                    "target_type='encoder_layer' requires output_hidden_states=True on "
                    "the config, but outputs.hidden_states is None. The __init__ should "
                    "have set this — check that the config wasn't overwritten."
                )
            target_full = outputs.hidden_states[self.target_layer_index + 1]
        else:
            target_full = sequence_input

        # Frame-level prediction loss.
        # Default (use_autoregressive_shift=True): position t's encoder output
        # predicts position t+1's target (next-target prediction — the NAPE
        # objective).
        # Ablation (use_autoregressive_shift=False): no shift — position t
        # predicts position t. This reduces the objective to an identity
        # mapping with no meaningful prediction signal and should fail to
        # converge.
        if self.use_autoregressive_shift:
            pred = sequence_output_proj[:, :-1, :]   # [B, T-1, P or D]
            target = target_full[:, 1:, :]            # [B, T-1, P or D]
        else:
            pred = sequence_output_proj              # [B, T, P or D]
            target = target_full                      # [B, T, P or D]

        if not self.disable_target_stop_gradient:
            target = target.detach()

        if self.loss_type == 'cosine':
            # Negative cosine similarity (NAPE standard)
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)
            frame_loss = -(pred_n * target_n).sum(dim=-1).mean()
        elif self.loss_type == 'mse':
            # MSE (LeWM standard / L2-squared)
            frame_loss = (pred - target).pow(2).mean()
        elif self.loss_type == 'l1':
            # Mean absolute error
            frame_loss = (pred - target).abs().mean()
        else:  # 'cross_entropy'
            # SimSiam-style cross-entropy similarity: treat each of the D channels
            # as a pseudo-category. Loss = -softmax(target) . log_softmax(pred),
            # i.e., cross-entropy between softmaxed pred and softmaxed target
            # distributions over the channel dim. Magnitude-invariant in the
            # absolute scale, but relative magnitudes shape the softmax sharpness.
            pred_logp = F.log_softmax(pred, dim=-1)
            target_p = F.softmax(target, dim=-1)
            frame_loss = -(target_p * pred_logp).sum(dim=-1).mean()

        total_loss = frame_loss

        return EmbeddedModelingOutput(
            loss=total_loss,
            frame_loss=frame_loss.detach(),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Classification head (for downstream fine-tuning)
# ---------------------------------------------------------------------------

class NapeForClassification(NapePreTrainedModel):
    def __init__(self, config: NapeConfig):
        super().__init__(config)
        self.num_patches = config.num_patches
        self.num_labels = config.num_labels

        # Pooling mode: "mean", "last_token", or "cls_token"
        # For backward compat, fall back to add_pooling_layer if pooling_mode not set
        if hasattr(config, 'pooling_mode'):
            self.pooling_mode = config.pooling_mode
        elif config.add_pooling_layer:
            self.pooling_mode = "mean"
        else:
            self.pooling_mode = "last_token"

        # Probe layer (linear probing): None → use last transformer layer (default;
        # matches standard fine-tuning). If set, use an intermediate layer or a
        # learned weighted combination of all layers as the feature source.
        # Options: None (default) | int K in [0, L-1] | "last" | "weighted_sum".
        self.probe_layer = getattr(config, 'probe_layer', None)
        # "last" is equivalent to the default (no probing) — read last_hidden_state
        # directly without paying the cost of exposing all hidden_states.
        if isinstance(self.probe_layer, str) and self.probe_layer == "last":
            self.probe_layer = None
        if self.probe_layer == "weighted_sum":
            # Learnable softmax weights over L+1 hidden states (embedding + all
            # transformer block outputs), plus a global scale (SUPERB convention).
            # Init so softmax puts essentially all the weight on the last layer at
            # start; the model then behaves like last-layer probing on epoch 1 and
            # only redistributes weight to earlier layers if that helps. This is
            # much more stable than uniform init, which mixes 13 differently-scaled
            # hidden states and often collapses on class-imbalanced tasks.
            init = torch.full((config.num_hidden_layers + 1,), -10.0)
            init[-1] = 0.0
            self.layer_weights = nn.Parameter(init)
            self.layer_scale = nn.Parameter(torch.tensor(1.0))
        elif isinstance(self.probe_layer, int):
            if not (0 <= self.probe_layer < config.num_hidden_layers):
                raise ValueError(
                    f"probe_layer={self.probe_layer} out of range "
                    f"[0, {config.num_hidden_layers - 1}] or 'weighted_sum'."
                )
        # The encoder inspects config.output_hidden_states directly (not a per-forward
        # kwarg), so we set it here at construction time whenever probing an
        # intermediate layer or building a weighted sum. For probe_layer=None
        # (or "last"), we skip this because last_hidden_state is always populated.
        if self.probe_layer is not None:
            config.output_hidden_states = True

        self.nape = NapeModel(config)

        # Always create fc_norm — applied uniformly to all pooling modes so the
        # pooling ablation compares like-for-like (mean vs cls vs last with the
        # same LayerNorm before the classifier).
        self.fc_norm = _make_norm(config.hidden_size, config)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    def forward(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ImageClassifierOutput:
        # If the encoder is fully frozen (linear probing), skip building the
        # autograd graph through it. This avoids storing activations for the
        # encoder — a large memory saving for intermediate probes / weighted-sum
        # probing, which otherwise keep all L+1 hidden states in memory.
        encoder_frozen = not any(p.requires_grad for p in self.nape.parameters())
        encoder_ctx = torch.no_grad() if encoder_frozen else contextlib.nullcontext()
        with encoder_ctx:
            outputs: BaseModelOutputWithEmbedding = self.nape(
                spectrogram,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        # Choose the feature source:
        #   None (default)      → last transformer block output (standard fine-tuning)
        #   int K               → output of the K-th transformer block (0-indexed)
        #   "weighted_sum"      → learned softmax-weighted combination of all layers
        #
        # NOTE: we deliberately do NOT apply self.nape.layernorm here.
        # That final LN was pretrained on layer-L statistics; feeding it a
        # layer-K distribution distorts it. Instead we rely on fc_norm (which
        # is fresh, trainable, and initialized as identity) to adapt to
        # whatever feature distribution the chosen layer produces.
        if self.probe_layer is None:
            sequence_output = outputs.last_hidden_state
        elif self.probe_layer == "weighted_sum":
            # outputs.hidden_states is a tuple: (L+1,) each [B, S, D].
            # index 0 = post-embedding input to first block; index i = output of block i-1.
            # With the layer_weights near-one-hot init on the last layer, the sum
            # is essentially the last hidden state at start; layer_weights and
            # layer_scale then learn to redistribute weight to other layers if it
            # helps. fc_norm handles the final normalization.
            stacked = torch.stack(outputs.hidden_states, dim=0)    # [L+1, B, S, D]
            w = F.softmax(self.layer_weights, dim=0)               # [L+1]
            sequence_output = (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)
            sequence_output = self.layer_scale * sequence_output
        else:
            # single-layer probe: hidden_states[K + 1] is the output of block K.
            sequence_output = outputs.hidden_states[self.probe_layer + 1]

        if self.pooling_mode == "mean":
            # Mean-pool over patch tokens (exclude CLS at position 0)
            patch_tokens = sequence_output[:, -self.num_patches:, :]
            pooled_output = patch_tokens.mean(dim=1)
            pooled_output = self.fc_norm(pooled_output)
        elif self.pooling_mode == "cls_token":
            # Use CLS token (position 0) — best when pretrained with CLS objective
            pooled_output = sequence_output[:, 0, :]
            pooled_output = self.fc_norm(pooled_output)
        else:
            # Use last token (most context in causal model)
            pooled_output = sequence_output[:, -1, :]
            pooled_output = self.fc_norm(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                loss = F.cross_entropy(logits, labels)

        # Only propagate hidden_states / attentions in the return if the caller
        # explicitly requested them via kwargs. When we're using hidden_states
        # internally for intermediate-layer probing, we do NOT want them in the
        # returned output — otherwise the HF Trainer's eval loop accumulates all
        # L+1 hidden states across every eval batch, causing OOM.
        caller_wants_hidden = kwargs.get("output_hidden_states", False)
        caller_wants_attn = kwargs.get("output_attentions", output_attentions if output_attentions is not None else False)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if caller_wants_hidden else None,
            attentions=outputs.attentions if caller_wants_attn else None,
        )


__all__ = [
    "NapeConfig",
    "NapeModel",
    "NapeForPreTraining",
    "NapeForClassification",
    "NapePreTrainedModel"
]
