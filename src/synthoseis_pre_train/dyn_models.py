# dyn_models.py

import math
from typing import List, Tuple
import torch
import torch.nn as nn

from .models import (
    ResBlock3d,
    AnisotropicResBlock3d,
    MambaBlock3d,
    _MAMBA_AVAILABLE,
)

# ---------------------------------------------------------------------------
# DynUNet-style planning
# ---------------------------------------------------------------------------

def _compute_dyn_plan(
    spatial_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    base_channels: int = 32,
    max_channels: int = 256,
    min_feature_size: int = 8,
) -> dict:
    """
    Compute DynUNet-style architecture plan.

    Returns:
        {
          "hidden_dims":   Tuple[int, ...],
          "strides":       List[Tuple[int,int,int]],
          "num_res_units": List[int],
          "deep_supervision_levels": List[int],  # decoder indices
        }
    """
    sz = list(spatial_size)
    sp = list(spacing)

    strides: List[Tuple[int, int, int]] = []
    hidden_dims: List[int] = []
    num_res_units: List[int] = []

    ch = base_channels
    level = 0

    while True:
        hidden_dims.append(ch)

        # heuristic: more blocks at high-res, fewer at low-res
        if level == 0:
            num_res_units.append(2)
        elif level == 1:
            num_res_units.append(2)
        else:
            num_res_units.append(1)

        # decide if we can downsample further
        can_down = [s >= 2 * min_feature_size for s in sz]
        if not any(can_down):
            break

        stride = []
        for i in range(3):
            if sz[i] >= 2 * min_feature_size:
                stride.append(2)
                sz[i] = math.floor(sz[i] / 2)
            else:
                stride.append(1)
        strides.append(tuple(stride))

        ch = min(ch * 2, max_channels)
        level += 1

    # deep supervision: all decoder levels with output size >= 1/4 of input
    # here we approximate by "all but the deepest two levels"
    n_levels = len(hidden_dims)
    deep_supervision_levels = [i for i in range(n_levels - 2)]

    return {
        "hidden_dims": tuple(hidden_dims),
        "strides": strides,
        "num_res_units": num_res_units,
        "deep_supervision_levels": deep_supervision_levels,
    }


def _build_res_block(block_type: str, in_ch: int, out_ch: int) -> nn.Module:
    if block_type == "resblock":
        return ResBlock3d(in_ch, out_ch)
    if block_type == "anisotropic":
        return AnisotropicResBlock3d(in_ch, out_ch)
    raise ValueError(f"Unsupported block_type: {block_type}")


# ---------------------------------------------------------------------------
# DynUNet-style encoder / decoder using your blocks
# ---------------------------------------------------------------------------

class DynUNetEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: Tuple[int, ...],
        strides: List[Tuple[int, int, int]],
        num_res_units: List[int],
        use_mamba: bool = False,
        block_type: str = "resblock",
        use_checkpoint: bool = True,
    ):
        super().__init__()
        from torch.utils.checkpoint import checkpoint as _grad_ckpt

        self.use_checkpoint = use_checkpoint
        self._ckpt = _grad_ckpt

        self.stem = _build_res_block(block_type, in_channels, hidden_dims[0])

        self.downs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for level in range(len(strides)):
            cin = hidden_dims[level]
            cout = hidden_dims[level + 1]
            stride = strides[level]

            self.downs.append(
                nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
            )

            stage_blocks = []
            for _ in range(num_res_units[level + 1]):
                if use_mamba:
                    stage_blocks.append(MambaBlock3d(cout))
                else:
                    stage_blocks.append(_build_res_block(block_type, cout, cout))
            self.blocks.append(nn.Sequential(*stage_blocks))

        self.bottleneck = (
            MambaBlock3d(hidden_dims[-1]) if use_mamba
            else _build_res_block(block_type, hidden_dims[-1], hidden_dims[-1])
        )

    def forward(self, x: torch.Tensor):
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        skips: List[torch.Tensor] = []

        if ckpt:
            x = self._ckpt(self.stem, x, use_reentrant=False)
        else:
            x = self.stem(x)
        skips.append(x)

        for down, block in zip(self.downs, self.blocks):
            x = down(x)
            if ckpt:
                x = self._ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
            skips.append(x)

        bot_in = skips.pop()
        if ckpt:
            x = self._ckpt(self.bottleneck, bot_in, use_reentrant=False)
        else:
            x = self.bottleneck(bot_in)
        return x, skips


class DynUNetDecoder3d(nn.Module):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        block_type: str = "resblock",
        use_checkpoint: bool = True,
    ):
        super().__init__()
        from torch.utils.checkpoint import checkpoint as _grad_ckpt

        self.use_checkpoint = use_checkpoint
        self._ckpt = _grad_ckpt

        dims = list(reversed(hidden_dims))
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(dims) - 1):
            deep_ch, skip_ch = dims[i], dims[i + 1]

            class _Up(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
                    self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)

                def forward(self, x):
                    x = self.up(x)
                    if x.shape[1] != self.conv.in_channels:
                        x = x[:, : self.conv.in_channels, ...]
                    return self.conv(x)

            self.ups.append(_Up(deep_ch, skip_ch))
            self.blocks.append(_build_res_block(block_type, 2 * skip_ch, skip_ch))

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for up, block, skip in zip(self.ups, self.blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            if ckpt:
                x = self._ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


# ---------------------------------------------------------------------------
# Complete DynUNet-style Seismic model
# ---------------------------------------------------------------------------

class DynSeismicUNet3d(nn.Module):
    HEAD_RECONSTRUCTION = "reconstruction"
    HEAD_SEGMENTATION = "segmentation"

    @staticmethod
    def _build_pre_head_block(mode: str, channels: int) -> nn.Module:
        mode = str(mode).strip().lower()
        if mode == "identity":
            return nn.Identity()
        if mode == "norm":
            return nn.InstanceNorm3d(channels, affine=True)
        if mode == "norm_gelu":
            return nn.Sequential(
                nn.InstanceNorm3d(channels, affine=True),
                nn.GELU(),
            )
        raise ValueError("pre_head_mode must be one of: identity, norm, norm_gelu")

    def __init__(
        self,
        input_channels: int = 1,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_mamba: bool = False,
        block_type: str = "resblock",
        use_checkpoint: bool = True,
        pre_head_mode: str = "norm_gelu",
    ):
        super().__init__()
        plan = _compute_dyn_plan(spatial_size=spatial_size, spacing=spacing)
        hidden_dims = plan["hidden_dims"]
        strides = plan["strides"]
        num_res_units = plan["num_res_units"]
        self.deep_supervision_levels = plan["deep_supervision_levels"]

        self.encoder = DynUNetEncoder3d(
            input_channels,
            hidden_dims,
            strides,
            num_res_units,
            use_mamba=use_mamba,
            block_type=block_type,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = DynUNetDecoder3d(
            hidden_dims,
            block_type=block_type,
            use_checkpoint=use_checkpoint,
        )

        self.pre_head_norm = self._build_pre_head_block(pre_head_mode, hidden_dims[0])
        self.head = nn.Conv3d(hidden_dims[0], input_channels, kernel_size=1)
        self._head_type = self.HEAD_RECONSTRUCTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return self.head(self.pre_head_norm(x)).float()

    def swap_to_segmentation_head(self, n_classes: int = 1, freeze_body: bool = True) -> None:
        in_ch = self.head.in_channels
        self.head = nn.Conv3d(in_ch, n_classes, kernel_size=1)
        self._head_type = self.HEAD_SEGMENTATION
        if freeze_body:
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                p.requires_grad = False

    def unfreeze_body(self) -> None:
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            p.requires_grad = True

    @property
    def head_type(self) -> str:
        return self._head_type

    @property
    def mamba_available(self) -> bool:
        return _MAMBA_AVAILABLE


def create_dyn_model(
    use_mamba: bool = False,
    block_type: str = "resblock",
    use_checkpoint: bool = True,
    **kwargs,
) -> DynSeismicUNet3d:
    """
    Factory mirroring create_model in models.py, but with DynUNet-style auto-scaling.

    kwargs: input_channels, spatial_size, spacing, ...
    """
    return DynSeismicUNet3d(
        use_mamba=use_mamba,
        block_type=block_type,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )
