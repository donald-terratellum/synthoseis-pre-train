"""
3D U-Net for Seismic Pre-training
====================================
Proper 3D U-Net with residual blocks and skip connections.

Supports two heads (swappable without reloading weights):
  - Reconstruction: linear output, trained with MSELoss on masked seismic
  - Segmentation:   logit output, trained with BCEWithLogitsLoss on fault labels

Optional U-Mamba upgrade (arXiv:2401.04722, Ma et al. 2024):
  Replaces encoder ResBlock3d with a hybrid CNN-SSM block.
  Requires CUDA + ``pip install mamba-ssm causal-conv1d``.
  Falls back silently to ResBlock3d on MPS/CPU.

Transfer learning workflow::

    # 1. Pre-train for reconstruction
    model = create_model(hidden_dims=(32, 64, 128, 256))
    # ... train with MSELoss ...

    # 2. Fine-tune encoder+decoder body for segmentation (head trains from scratch)
    model.swap_to_segmentation_head(n_classes=1, freeze_body=True)
    # ... fine-tune with BCEWithLogitsLoss ...

    # 3. Optionally unfreeze everything for end-to-end fine-tuning
    model.unfreeze_body()
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Residual building block
# ---------------------------------------------------------------------------

class ResBlock3d(nn.Module):
    """Two Conv3d layers with a residual skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.GELU()
        self.proj = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


# ---------------------------------------------------------------------------
# Optional U-Mamba block (arXiv:2401.04722)
# ---------------------------------------------------------------------------

_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba  # type: ignore
    _MAMBA_AVAILABLE = True
except ImportError:
    pass


class MambaBlock3d(nn.Module):
    """
    Hybrid CNN-SSM block based on U-Mamba (arXiv:2401.04722).

    Architecture:
      - Depthwise Conv3d  → local spatial features  (CNN branch)
      - Mamba SSM         → long-range dependencies  (SSM branch)
      - Element-wise add of both branches + input residual

    Requires CUDA + ``pip install mamba-ssm causal-conv1d``.
    Falls back to ResBlock3d automatically if mamba_ssm is not installed.
    """

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            self._block = ResBlock3d(channels, channels)
            self._use_mamba = False
            return

        self._use_mamba = True
        # Local CNN branch (depthwise)
        self.dw_conv = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_norm = nn.InstanceNorm3d(channels, affine=True)
        # SSM branch
        self.seq_norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        # Output
        self.out_norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_mamba:
            return self._block(x)

        B, C, D, H, W = x.shape
        # CNN branch
        cnn = self.act(self.dw_norm(self.dw_conv(x)))
        # SSM branch: (B,C,D,H,W) → (B, D*H*W, C) → Mamba → (B,C,D,H,W)
        seq = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        seq = self.mamba(self.seq_norm(seq))
        ssm = seq.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return self.act(self.out_norm(cnn + ssm + x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class UNetEncoder3d(nn.Module):
    """
    3D U-Net encoder.

    Notation: (B, C_channels, S_spatial) — S is shorthand for S×S×S voxels.
    All spatial sizes are cubic. Input and output are always the same spatial shape.

    Channel plan for hidden_dims=(32, 64, 128, 256) and input (B,C=1,S=128):
      stem    : (B, C=  1, S=128) → (B, C= 32, S=128)   ← skip[0]  537 MB each @ B=2
      down+enc: (B, C= 32, S=128) → (B, C= 64, S= 64)   ← skip[1]  134 MB each @ B=2
      down+enc: (B, C= 64, S= 64) → (B, C=128, S= 32)   ← skip[2]   34 MB each @ B=2
      down+enc: (B, C=128, S= 32) → (B, C=256, S= 16)   → bottleneck (B,C=256,S=16)

    Returns bottleneck tensor and skips list [skip[0], skip[1], skip[2]].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Tuple[int, ...],
        use_mamba: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stem = ResBlock3d(in_channels, hidden_dims[0])

        self.downsamples = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            cin, cout = hidden_dims[i], hidden_dims[i + 1]
            self.downsamples.append(nn.Conv3d(cin, cout, kernel_size=2, stride=2, bias=False))
            self.enc_blocks.append(MambaBlock3d(cout) if use_mamba else ResBlock3d(cout, cout))

        bot_ch = hidden_dims[-1]
        self.bottleneck = MambaBlock3d(bot_ch) if use_mamba else ResBlock3d(bot_ch, bot_ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        skips: List[torch.Tensor] = []
        x = _grad_ckpt(self.stem, x, use_reentrant=False) if ckpt else self.stem(x)
        skips.append(x)

        for down, block in zip(self.downsamples, self.enc_blocks):
            x = down(x)
            x = _grad_ckpt(block, x, use_reentrant=False) if ckpt else block(x)
            skips.append(x)

        # Deepest entry goes through bottleneck; removed from skips list
        bot_in = skips.pop()
        x = _grad_ckpt(self.bottleneck, bot_in, use_reentrant=False) if ckpt else self.bottleneck(bot_in)
        return x, skips  # skips: [shallow, ..., second-deepest]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetDecoder3d(nn.Module):
    """
    3D U-Net decoder with skip connections.

    Notation: (B, C_channels, S_spatial) — S is shorthand for S×S×S voxels.
    Spatial size doubles at each upsample; skip channel counts must match encoder.

    Channel plan for hidden_dims=(32,64,128,256), bottleneck (B,C=256,S=16):
      up+cat+block: up→(B,C=128,S=32) cat skip(B,C=128,S=32) → (B,C=256,S=32) → (B,C=128,S=32)
      up+cat+block: up→(B,C= 64,S=64) cat skip(B,C= 64,S=64) → (B,C=128,S=64) → (B,C= 64,S=64)
      up+cat+block: up→(B,C= 32,S=128) cat skip(B,C=32,S=128) → (B,C=64,S=128) → (B,C=32,S=128)
    Output spatial size = input spatial size (128×128×128 in, 128×128×128 out).
    """

    def __init__(self, hidden_dims: Tuple[int, ...], use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dims = list(reversed(hidden_dims))  # e.g. [256, 128, 64, 32]
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            deep_ch, skip_ch = dims[i], dims[i + 1]
            # Replace transposed conv upsampling (checkerboard artifacts)
            # with artifact-free upsample (trilinear) followed by a Conv3d.
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(deep_ch, skip_ch, kernel_size=3, padding=1, bias=False),
                )
            )
            self.dec_blocks.append(ResBlock3d(2 * skip_ch, skip_ch))

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for up, block, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = _grad_ckpt(block, x, use_reentrant=False) if ckpt else block(x)
        return x


# ---------------------------------------------------------------------------
# Complete model with swappable head
# ---------------------------------------------------------------------------

class SeismicUNet3d(nn.Module):
    """
    3D U-Net for seismic reconstruction pre-training and segmentation fine-tuning.

    Default head produces a linear reconstruction output (use with MSELoss).
    Call ``swap_to_segmentation_head`` to replace it with a logit head for
    fine-tuning on fault/lithology labels (use with BCEWithLogitsLoss).

    All encoder + decoder weights are preserved across the swap; only the
    final 1×1×1 Conv3d is replaced.
    """

    HEAD_RECONSTRUCTION = "reconstruction"
    HEAD_SEGMENTATION = "segmentation"

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        use_mamba: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.encoder = UNetEncoder3d(input_channels, hidden_dims, use_mamba, use_checkpoint)
        self.decoder = UNetDecoder3d(hidden_dims, use_checkpoint)
        self.head = nn.Conv3d(hidden_dims[0], input_channels, kernel_size=1)
        self._head_type = self.HEAD_RECONSTRUCTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return self.head(x).float()

    def swap_to_segmentation_head(
        self,
        n_classes: int = 1,
        freeze_body: bool = True,
    ) -> None:
        """
        Replace reconstruction head with a segmentation head.

        Args:
            n_classes: Output classes. 1 = binary (e.g. faults) → BCEWithLogitsLoss.
            freeze_body: Freeze encoder + decoder so only the new head trains initially.
                         Call ``unfreeze_body()`` for full end-to-end fine-tuning later.
        """
        in_ch = self.head.in_channels
        self.head = nn.Conv3d(in_ch, n_classes, kernel_size=1)
        self._head_type = self.HEAD_SEGMENTATION
        if freeze_body:
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                p.requires_grad = False

    def unfreeze_body(self) -> None:
        """Re-enable gradient flow through encoder and decoder."""
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            p.requires_grad = True

    @property
    def head_type(self) -> str:
        return self._head_type

    @property
    def mamba_available(self) -> bool:
        return _MAMBA_AVAILABLE


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    use_mamba: bool = False,
    use_checkpoint: bool = True,
    **kwargs,
) -> SeismicUNet3d:
    """
    Create a SeismicUNet3d model.

    Args:
        use_mamba: Use MambaBlock3d in encoder stages (U-Mamba, arXiv:2401.04722).
                   Requires CUDA + ``pip install mamba-ssm causal-conv1d``.
                   Falls back to ResBlock3d on MPS/CPU.
        use_checkpoint: Use gradient checkpointing in encoder/decoder ResBlocks.
                   Trades ~3-4x less activation memory for ~20% slower training
                   (backward recomputes each block's forward pass).
                   Default True; disable only if you have memory to spare.
        **kwargs:  Forwarded to SeismicUNet3d:
                   ``input_channels``, ``hidden_dims``, ``spatial_size``.
    """
    return SeismicUNet3d(use_mamba=use_mamba, use_checkpoint=use_checkpoint, **kwargs)
