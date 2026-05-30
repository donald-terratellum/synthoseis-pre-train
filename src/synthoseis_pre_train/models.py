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
from typing import List, Tuple, cast


def _same_padding_3d(kernel_size: int) -> int:
    """Return symmetric padding for stride-1 odd kernels that preserves shape."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    return kernel_size // 2


def _resolve_stage_kernel_sizes(hidden_dims: Tuple[int, ...], kernel_sizes: Tuple[int, ...] | None) -> Tuple[int, ...]:
    """Return one odd kernel size per encoder stage (same length as hidden_dims)."""
    if kernel_sizes is None:
        return tuple(3 for _ in hidden_dims)
    if len(kernel_sizes) != len(hidden_dims):
        raise ValueError(
            "kernel_sizes must have the same length as hidden_dims "
            f"(got {len(kernel_sizes)} vs {len(hidden_dims)})"
        )
    for k in kernel_sizes:
        _same_padding_3d(k)  # validates odd positive kernel sizes
    return kernel_sizes


# ---------------------------------------------------------------------------
# Residual building block
# ---------------------------------------------------------------------------

class ResBlock3d(nn.Module):
    """Two Conv3d layers with a residual skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = _same_padding_3d(kernel_size)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
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
    Mamba = None  # type: ignore[assignment]


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

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = _same_padding_3d(kernel_size)
        if not _MAMBA_AVAILABLE:
            self._block = ResBlock3d(channels, channels, kernel_size=kernel_size)
            self._use_mamba = False
            return

        self._use_mamba = True
        # Local CNN branch (depthwise)
        self.dw_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.dw_norm = nn.InstanceNorm3d(channels, affine=True)
        # SSM branch
        self.seq_norm = nn.LayerNorm(channels)
        assert Mamba is not None
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
        kernel_sizes: Tuple[int, ...] | None = None,
        use_mamba: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        stage_kernels = _resolve_stage_kernel_sizes(hidden_dims, kernel_sizes)
        self.stem = ResBlock3d(in_channels, hidden_dims[0], kernel_size=stage_kernels[0])

        self.downsamples = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            cin, cout = hidden_dims[i], hidden_dims[i + 1]
            self.downsamples.append(nn.Conv3d(cin, cout, kernel_size=2, stride=2, bias=False))
            block_kernel = stage_kernels[i + 1]
            self.enc_blocks.append(
                MambaBlock3d(cout, kernel_size=block_kernel)
                if use_mamba
                else ResBlock3d(cout, cout, kernel_size=block_kernel)
            )

        bot_ch = hidden_dims[-1]
        # Keep bottleneck conservative; most receptive-field growth happens from depth.
        self.bottleneck = MambaBlock3d(bot_ch, kernel_size=3) if use_mamba else ResBlock3d(bot_ch, bot_ch, kernel_size=3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        skips: List[torch.Tensor] = []
        x = cast(torch.Tensor, _grad_ckpt(self.stem, x, use_reentrant=False)) if ckpt else self.stem(x)
        skips.append(x)

        for down, block in zip(self.downsamples, self.enc_blocks):
            x = down(x)
            x = cast(torch.Tensor, _grad_ckpt(block, x, use_reentrant=False)) if ckpt else block(x)
            skips.append(x)

        # Deepest entry goes through bottleneck; removed from skips list
        bot_in = skips.pop()
        x = cast(torch.Tensor, _grad_ckpt(self.bottleneck, bot_in, use_reentrant=False)) if ckpt else self.bottleneck(bot_in)
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

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...] | None = None,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        stage_kernels = _resolve_stage_kernel_sizes(hidden_dims, kernel_sizes)
        dims = list(reversed(hidden_dims))  # e.g. [256, 128, 64, 32]
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            deep_ch, skip_ch = dims[i], dims[i + 1]
            self.upsamples.append(
                nn.ConvTranspose3d(deep_ch, skip_ch, kernel_size=2, stride=2, bias=False)
            )
            # Decoder kernels mirror encoder scale kernels from deep->shallow.
            self.dec_blocks.append(ResBlock3d(2 * skip_ch, skip_ch, kernel_size=stage_kernels[-(i + 2)]))

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for up, block, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = cast(torch.Tensor, _grad_ckpt(block, x, use_reentrant=False)) if ckpt else block(x)
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
        kernel_sizes: Tuple[int, ...] | None = None,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        use_mamba: bool = False,
        use_checkpoint: bool = True,
        deep_reconstruction_head: bool = False,
    ):
        super().__init__()
        self.encoder = UNetEncoder3d(input_channels, hidden_dims, kernel_sizes, use_mamba, use_checkpoint)
        self.decoder = UNetDecoder3d(hidden_dims, kernel_sizes, use_checkpoint)
        if deep_reconstruction_head:
            # Two Conv3d layers with norm and activation between
            in_ch = hidden_dims[0]
            mid_ch = max(1, in_ch // 2)
            self.head = nn.Sequential(
                nn.Conv3d(in_ch, mid_ch, kernel_size=1),
                nn.InstanceNorm3d(mid_ch, affine=True),
                nn.GELU(),
                nn.Conv3d(mid_ch, input_channels, kernel_size=1),
            )
        else:
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
    kernel_sizes: Tuple[int, ...] | None = None,
    deep_reconstruction_head: bool = False,
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
        kernel_sizes: Optional per-scale odd kernel schedule with same length as
                   ``hidden_dims`` (e.g., (7, 5, 3, 3) for (16, 32, 64, 128)).
                   Larger kernels are applied in shallow encoder/decoder stages.
                   Bottleneck remains fixed at 3.
        **kwargs:  Forwarded to SeismicUNet3d:
                   ``input_channels``, ``hidden_dims``, ``spatial_size``.
    """
    return SeismicUNet3d(
        use_mamba=use_mamba,
        use_checkpoint=use_checkpoint,
        kernel_sizes=kernel_sizes,
        deep_reconstruction_head=deep_reconstruction_head,
        **kwargs,
    )


def report_masked_voxel_stats(input: torch.Tensor):
    """
    Print detailed masking/extrema/trace stats for a 3D seismic batch.
    Args:
        input: (B, C, Z, X, Y) tensor (or (Z, X, Y) if squeezed)
        extrema_mask: (B, C, Z, X, Y) or (Z, X, Y) or None
    """
    # Squeeze batch/channel if present
    while input.ndim > 3:
        input = input[0]
    Z, X, Y = input.shape
    total_voxels = Z * X * Y
    # 1. Container (nonzero bounding box)
    nonzero = (input != 0)
    nz_idx = nonzero.nonzero(as_tuple=False)
    if nz_idx.numel() == 0:
        print(". retained percentages: (container, extrema, clustered) = (0.00%, 0.00%, 0.00%) --> 0.00% retained")
        return
    # Find first and last nonzero indices in each dimension (robust, efficient)
    z_nonzero = (input != 0).any(dim=(1, 2))
    x_nonzero = (input != 0).any(dim=(0, 2))
    y_nonzero = (input != 0).any(dim=(0, 1))
    z_indices = z_nonzero.nonzero(as_tuple=False).squeeze()
    x_indices = x_nonzero.nonzero(as_tuple=False).squeeze()
    y_indices = y_nonzero.nonzero(as_tuple=False).squeeze()
    if z_indices.numel() == 0 or x_indices.numel() == 0 or y_indices.numel() == 0:
        print(". retained percentages: (container, extrema, clustered) = (0.00%, 0.00%, 0.00%) --> 0.00% retained")
        return
    min_z, max_z = z_indices[0].item(), z_indices[-1].item()
    min_x, max_x = x_indices[0].item(), x_indices[-1].item()
    min_y, max_y = y_indices[0].item(), y_indices[-1].item()
    z_valid, x_valid, y_valid = tuple((max_z - min_z + 1, max_x - min_x + 1, max_y - min_y + 1))
    valid_size = z_valid * x_valid * y_valid
    container_pct = 100.0 * valid_size / total_voxels
    # 2. Extrema percent (if provided, using per-trace nonzero count method)
    extrema_valid = input[min_z:max_z+1, min_x:max_x+1, min_y:max_y+1]
    # Step 1: count nonzero values along z for each (x, y)
    extrema_trace_counts = (extrema_valid != 0).sum(dim=0)  # shape (X, Y)
    # Step 2: count number of traces with any extrema
    nonzero_traces = (extrema_trace_counts > 0)
    num_traces = nonzero_traces.sum().item()
    depth = max_z - min_z + 1
    denom = num_traces * depth if depth > 0 else 1
    numer = extrema_trace_counts.sum().item()
    extrema_pct = 100.0 * numer / denom if denom > 0 else 0.0

    # 3. Clustered traces (fully zeroed traces along Z)
    # For each (x, y), check if all Z are zero
    # trace_zero = (output == 0).all(dim=0)  # shape (X, Y)
    # clustered_count = trace_zero.count_nonzero().item()
    clustered_pct = 100.0 * num_traces / ((max_x - min_x + 1) * (max_y - min_y + 1))
    retained_pct = 100.0 * nz_idx.numel() / total_voxels
    # Print
    print(
        f"         . retained percentages: (container shape, container, extrema, clustered) = ("\
        f"{(z_valid, x_valid, y_valid)}, "\
        f"{container_pct:.2f}%, "\
        f"{extrema_pct:.2f}%, "\
        f"{clustered_pct:.2f}%) --> "\
        f"{retained_pct:.2f}% retained"
    )
