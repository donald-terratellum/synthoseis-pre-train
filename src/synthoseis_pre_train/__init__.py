"""
Seismic 3D Mamba Pre-training Repository
==========================================

This repository contains code for pre-training 3D Mamba-based models
for seismic data reconstruction from masked inputs.

Key components:
- Masking strategies for seismic data (peaks, troughs, trace masking)
- Data augmentation (stretch/squeeze, time-to-depth conversion simulation)
- Integration with 3D Mamba vision models
"""

__version__ = "0.1.0"

from .transforms import (
	QuantileNormalConfig,
	QuantileNormalTransform,
	ensure_quantile_normal_transform,
	load_quantile_normal_transform,
)

__all__ = [
	"QuantileNormalConfig",
	"QuantileNormalTransform",
	"ensure_quantile_normal_transform",
	"load_quantile_normal_transform",
]