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
"""Nape model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NapeConfig(PretrainedConfig):
    r"""
    Configuration class for Nape.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (feed-forward) layer in the Transformer encoder.
        use_gated_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use a gated MLP (SwiGLU-style) instead of a standard feed-forward block.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        rope_theta (`float`, *optional*, defaults to 100.0):
            Base period used for rotary positional embeddings.
        freq_bins (`int`, *optional*, defaults to 128):
            Number of frequency bins (mel bins) in the input spectrogram.
        target_time_frames (`int`, *optional*, defaults to 1008):
            Number of time frames in the input spectrogram after padding.
            Must be divisible by patch_size. For 10s audio at 16kHz with 10ms hop,
            raw frames ~1000, padded to 1008 (divisible by 16).
        patch_size (`int`, *optional*, defaults to 16):
            The size of each square patch for both frequency and time dimensions.
        num_channels (`int`, *optional*, defaults to 1):
            The number of input channels (1 for single-channel mel spectrogram).
        patch_order (`str`, *optional*, defaults to `"raster"`):
            The order in which 2D patches are serialized into a 1D sequence.
            "raster": row-major order (all time steps for freq band 0, then freq band 1, ...).
            "time_major": column-major order (all freq bands at time 0, then time 1, ...).
            "zigzag": like raster but alternating direction per row (row 0 left-to-right,
                row 1 right-to-left, etc.), preserving spatial locality at row transitions.
            "diagonal": anti-diagonal scan, sorted by (freq + time) index. Provides
                balance context in both frequency and time dimensions.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to the query and key projections before attention.
        qk_norm_bias (`bool`, *optional*, defaults to `False`):
            Whether the query/key normalization layers use a bias term.
        qk_norm_affine (`bool`, *optional*, defaults to `False`):
            Whether the query/key normalization layers use learnable affine parameters.
        layerscale_value (`float`, *optional*, defaults to 1e-5):
            Initial value for LayerScale factors.
        drop_path_prob (`float`, *optional*, defaults to 0.0):
            Stochastic depth (DropPath) rate used in the encoder blocks.
        add_pooling_layer (`bool`, *optional*, defaults to `False`):
            Whether to add a mean-pooling layer for classification (otherwise use last token).
        is_causal (`bool`, *optional*, defaults to `True`):
            Whether to use a causal attention mask (for autoregressive-style training).
            Setting to `False` disables the causal mask — ablation for the NAPE causal
            objective.
        norm_type (`str`, *optional*, defaults to `"layernorm"`):
            Normalization layer used throughout the encoder (pre-norms, final norm,
            QK-Norm if enabled). Options: "layernorm" or "rmsnorm". RMSNorm uses
            `torch.nn.RMSNorm` if available (PyTorch >= 2.4) and a manual fallback
            otherwise.
        position_embedding_type (`str`, *optional*, defaults to `"rope"`):
            How positional information is injected. Options:
              - "rope"     : 2D rotary positional embeddings applied inside attention
                             (default; uses `NapeRopePositionEmbedding`).
              - "absolute" : Learnable absolute positional embeddings added to the
                             patch embeddings before the encoder. Sized to cover
                             [CLS, register tokens, patches].
        sample_rate (`int`, *optional*, defaults to 16000):
            Audio sample rate in Hz. Used for reference; spectrogram computation happens in the data pipeline.
        hop_length (`int`, *optional*, defaults to 160):
            Hop length for STFT (10ms at 16kHz).
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel filterbank channels.
        audio_duration (`float`, *optional*, defaults to 10.0):
            Target audio clip duration in seconds.
        use_prediction_head (`bool`, *optional*, defaults to `False`):
            Whether to use an MLP prediction head (projector) between the transformer
            output and the prediction loss during pretraining. Inspired by SimSiam,
            this separates the representation space from the prediction space.
        prediction_head_type (`str`, *optional*, defaults to `"mlp2"`):
            Architecture of the prediction head when `use_prediction_head=True`.
            Options:
              - "mlp2"    : 2-layer MLP (Linear -> GELU -> Linear). The original
                            simple head.
              - "simsiam" : 3-layer MLP with intermediate normalization
                            (Linear -> Norm -> GELU -> Linear -> Norm -> GELU -> Linear).
                            Final norm is omitted per SimSiam (Chen & He, 2020),
                            which reports it causes training instability.
              - "transformer": Transformer predictor in the style of JEPA-family methods: a 2-
                                layer causal Transformer (16 heads) operating in the encoder’s hidden dimension.
        loss_type (`str`, *optional*, defaults to `"cosine"`):
            Type of pretraining loss.
              - "cosine":        negative cosine similarity (standard NAPE, fully
                                 magnitude-invariant). Default.
              - "mse":           mean squared error / L2-squared, equivalent to
                                 LeWM/JEPA-style. Prone to magnitude collapse without
                                 additional anti-collapse (EMA, SIGReg, ...).
              - "l1":            mean absolute error, more robust to outliers than MSE.
                                 Same collapse risk as MSE.
              - "cross_entropy": SimSiam-style cross-entropy similarity
                                 -softmax(target) . log_softmax(pred), over the channel
                                 dim. Treats each of the D channels as a "pseudo-class."
                                 Magnitude-invariant in absolute scale; relative
                                 magnitudes still shape the softmax sharpness.
        target_type (`str`, *optional*, defaults to `"patch_embedding"`):
            What the prediction loss targets.
              - "patch_embedding": output of the patch embedding layer (standard NAPE target).
              - "raw_mel": raw mel-spectrogram patch values (patch_size**2 * num_channels per
                patch). With this setting the predictor output is projected from hidden_size
                down to the raw patch dimensionality via an additional Linear layer.
              - "encoder_layer": output of a specific encoder layer (deep-feature target).
                Which layer is controlled by `target_layer_index`. WARNING: stop-gradient
                alone may not prevent representational collapse since both prediction
                and target come from the same encoder. For non-collapsing variants you
                need EMA or SIGReg in addition.
        target_layer_index (`int`, *optional*, defaults to `num_hidden_layers // 2`):
            Which encoder layer's output to use as the target when target_type is
            "encoder_layer". 0-indexed where 0 = output of the first transformer block
            and num_hidden_layers - 1 = output of the last block (pre-final-LayerNorm).
            Ignored unless target_type='encoder_layer'.
        raw_mel_normalize (`bool`, *optional*, defaults to `True`):
            Whether to apply per-patch normalization to the raw mel target (subtract patch
            mean, divide by patch std). MAE-style. Only used when target_type='raw_mel'.
        patch_embed_type (`str`, *optional*, defaults to `"conv2d"`):
            Patchifier architecture.
              - "conv2d": single non-overlapping Conv2d (kernel=stride=patch_size). Default.
                Produces a 2D token grid (freq × time) — standard ViT convention.
              - "convstem": 4-layer 3x3 stride-2 conv stack (ConvViT-style). Total spatial
                reduction is still 16x; channels grow 1 → D/16 → D/8 → D/4 → D, with
                BatchNorm and GELU between layers. 2D token grid. Requires patch_size=16.
              - "speech_stem": speech-encoder-style stack (Gemma-3 inspired). 2 conv layers
                with 3x3 kernels and channel progression 1 → 128 → 32, downsampling time by
                `speech_stem_downsample` (default 2). Frequency is flattened into channels.
                Produces a 1D time-indexed token sequence with no frequency split — each
                token represents all freq bins at one downsampled time step. The
                downstream RoPE handles this as a (1 × T/downsample) grid.
        speech_stem_downsample (`int`, *optional*, defaults to `2`):
            Temporal downsampling factor for `patch_embed_type='speech_stem'`. With
            target_time_frames=1008 and downsample=2, the output sequence length is
            504 — matching the default conv2d patchifier's sequence length for
            apples-to-apples ablations. Use 4 for Gemma-3's full ratio (sequence of
            252) at the cost of less temporal resolution.
        use_autoregressive_shift (`bool`, *optional*, defaults to `True`):
            Whether to apply the autoregressive shift in the prediction loss. Default
            True predicts position t+1's patch embedding from position t's encoder
            output (standard NAPE objective). Setting False removes the shift, reducing
            the objective to an identity mapping with no meaningful prediction signal;
            used as the no-shift ablation from NAPE Table 1a.
        disable_target_stop_gradient (`bool`, *optional*, defaults to `False`):
            By default the target embedding in the prediction loss is detached
            (stop-gradient), which is the standard NEPA / I-JEPA recipe and the
            primary anti-collapse mechanism. Setting this to True lets gradient
            flow through both sides of the prediction loss (LeWM-style); in that
            regime an auxiliary regularizer such as SIGReg is required to
            prevent representational collapse.
        mask_ratio (`float`, *optional*, defaults to 0.0):
            Fraction of patch embeddings to randomly mask during pretraining.
            Masked patches are replaced with a learnable mask token, while the
            prediction target remains the clean (unmasked) embedding. Loss is
            computed over all positions (both masked and unmasked). 0.0 = no masking
            (standard NAPE). Typical values: 0.15-0.30.
        num_register_tokens (`int`, *optional*, defaults to 0):
            Number of learnable register tokens to insert after CLS and before patches.
            Registers act as global information aggregation slots with no positional encoding.
            Inspired by DINOv2-reg (Darcet et al., 2024). Typically used during fine-tuning only.

    Example:

    ```python
    >>> from models.nape import NapeConfig, NapeModel

    >>> configuration = NapeConfig()
    >>> model = NapeModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "nape"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_gated_mlp=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        rope_theta=100.0,
        freq_bins=128,
        target_time_frames=1008,
        patch_size=16,
        num_channels=1,
        patch_order="raster",
        qkv_bias=True,
        qk_norm=False,
        qk_norm_bias=False,
        qk_norm_affine=False,
        layerscale_value=1e-5,
        drop_path_prob=0.0,
        add_pooling_layer=False,
        is_causal=True,
        # Architecture ablation knobs
        norm_type: str = "layernorm",
        position_embedding_type: str = "rope",
        # Audio preprocessing parameters (stored for reference / reproducibility)
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_mels: int = 128,
        audio_duration: float = 10.0,
        # Pretraining objective
        use_prediction_head: bool = False,
        prediction_head_type: str = "mlp2",
        loss_type: str = "cosine",
        target_type: str = "patch_embedding",
        target_layer_index: int = None,
        raw_mel_normalize: bool = True,
        patch_embed_type: str = "conv2d",
        speech_stem_downsample: int = 2,
        use_autoregressive_shift: bool = True,
        disable_target_stop_gradient: bool = False,
        mask_ratio: float = 0.0,
        num_register_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.use_gated_mlp = use_gated_mlp
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.freq_bins = freq_bins
        self.target_time_frames = target_time_frames
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_order = patch_order
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.qk_norm_bias = qk_norm_bias
        self.qk_norm_affine = qk_norm_affine
        self.layerscale_value = layerscale_value
        self.drop_path_prob = drop_path_prob
        self.add_pooling_layer = add_pooling_layer
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.position_embedding_type = position_embedding_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.audio_duration = audio_duration
        self.use_prediction_head = use_prediction_head
        self.prediction_head_type = prediction_head_type
        self.loss_type = loss_type
        self.target_type = target_type
        self.target_layer_index = target_layer_index
        self.raw_mel_normalize = raw_mel_normalize
        self.patch_embed_type = patch_embed_type
        self.speech_stem_downsample = speech_stem_downsample
        self.use_autoregressive_shift = use_autoregressive_shift
        self.disable_target_stop_gradient = disable_target_stop_gradient
        self.mask_ratio = mask_ratio
        self.num_register_tokens = num_register_tokens

        # Validate patch divisibility
        if self.freq_bins % self.patch_size != 0:
            raise ValueError(
                f"freq_bins ({self.freq_bins}) must be divisible by patch_size ({self.patch_size})."
            )
        if self.target_time_frames % self.patch_size != 0:
            raise ValueError(
                f"target_time_frames ({self.target_time_frames}) must be divisible by patch_size ({self.patch_size})."
            )
        if self.patch_order not in ("raster", "time_major", "zigzag", "diagonal"):
            raise ValueError(
                f"patch_order must be 'raster', 'time_major', 'zigzag', or 'diagonal', got '{self.patch_order}'."
            )

    @property
    def num_patches_freq(self):
        return self.freq_bins // self.patch_size

    @property
    def num_patches_time(self):
        return self.target_time_frames // self.patch_size

    @property
    def num_patches(self):
        return self.num_patches_freq * self.num_patches_time


__all__ = ["NapeConfig"]
