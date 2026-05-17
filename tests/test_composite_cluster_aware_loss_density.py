#!/usr/bin/env python3
"""Test cases for CompositeClusterAwareLoss with masking density-based weighting."""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss


class TestDensityBasedClusterAwareLoss:
    """Test suite for density-based cluster-aware weighting.
    
    The redesigned CompositeClusterAwareLoss uses masking DENSITY to identify
    regions:
    - base_mask: traces with LOW masking density (below median)
    - cluster_mask: traces with HIGH masking density (above median)
    
    This works with sparse voxel-level masking where individual voxels are
    masked within traces, not requiring completely-masked traces.
    """
    
    def test_traces_partition_by_masking_density(self):
        """Verify that base and cluster masks partition traces at median density."""
        B, C, D, H, W = 1, 1, 32, 16, 16
        reference = torch.randn(B, C, D, H, W)
        
        # Create a mask: left half has low density, right half has high density
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :, :, :8] = 1.0   # Left: 100% valid (0% masked)
        valid_mask[:, :, :, :, 8:] = 0.0   # Right: 0% valid (100% masked)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, valid_mask)
        
        # Check shapes
        assert base_mask.shape == (B, C, D, H, W)
        assert cluster_mask.shape == (B, C, D, H, W)
        
        # base_mask + cluster_mask should partition all traces (no overlap, complete coverage)
        spatial_sum = base_mask[:, 0, 0, :, :] + cluster_mask[:, 0, 0, :, :]
        assert torch.allclose(spatial_sum, torch.ones_like(spatial_sum)), \
            "base and cluster masks should partition all traces"
        
        # Left half (low density) should be predominantly base
        left_base_ratio = base_mask[0, 0, 0, :, :8].mean().item()
        assert left_base_ratio > 0.5, f"Left (low-density) traces should be mostly base, got {left_base_ratio}"
        
        # Right half (high density) should be predominantly cluster
        right_cluster_ratio = cluster_mask[0, 0, 0, :, 8:].mean().item()
        assert right_cluster_ratio > 0.5, f"Right (high-density) traces should be mostly cluster, got {right_cluster_ratio}"
        
        print("✓ Traces correctly partitioned by masking density")
    
    def test_weight_maps_correct_ranges(self):
        """Verify diagnostic weight maps have correct shapes and value ranges."""
        B, C, D, H, W = 2, 1, 24, 12, 12
        reference = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :8, :, :] = 0.5  # Partial masking in first half
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(
            base_loss,
            base_weight=1.0 / 3.0,
            cluster_weight=2.0 / 3.0
        )
        
        base_weight_map, cluster_weight_map = criterion.diagnostic_weight_maps(reference, valid_mask)
        
        # Check shapes
        assert base_weight_map.shape == (B, C, D, H, W)
        assert cluster_weight_map.shape == (B, C, D, H, W)
        
        # Check value ranges
        base_max = criterion.base_weight * 1.01  # Small tolerance for numerical error
        cluster_max = criterion.cluster_weight * 1.01
        
        assert base_weight_map.min() >= -1e-6, "base_weight_map should have non-negative min"
        assert base_weight_map.max() <= base_max, f"base_weight_map max {base_weight_map.max()} > {base_max}"
        
        assert cluster_weight_map.min() >= -1e-6, "cluster_weight_map should have non-negative min"
        assert cluster_weight_map.max() <= cluster_max, f"cluster_weight_map max {cluster_weight_map.max()} > {cluster_max}"
        
        print("✓ Weight maps have correct shapes and value ranges")
    
    def test_no_mask_gives_uniform_base_weighting(self):
        """Verify that omitting valid_mask results in uniform base weighting."""
        B, C, D, H, W = 1, 1, 16, 8, 8
        reference = torch.randn(B, C, D, H, W)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, None)
        
        # Without a mask, all voxels should get base weighting
        assert torch.allclose(base_mask, torch.ones_like(base_mask)), \
            "Without mask, all voxels should be in base region"
        assert torch.allclose(cluster_mask, torch.zeros_like(cluster_mask)), \
            "Without mask, no voxels should be in cluster region"
        
        print("✓ No mask correctly defaults to uniform base weighting")
    
    def test_forward_pass_produces_valid_loss(self):
        """Verify forward pass produces valid scalar loss without NaN/Inf."""
        B, C, D, H, W = 2, 1, 32, 12, 12
        pred = torch.randn(B, C, D, H, W)
        target = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[0, :, :16, :, :] = 0.5  # Partial masking in first batch
        valid_mask[1, :, :, :6, :] = 0.2   # Different masking pattern in second batch
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        loss = criterion(pred, target, valid_mask)
        
        # Check it's a scalar
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got shape {loss.shape}"
        assert not torch.isnan(loss).any(), "Loss should not be NaN"
        assert not torch.isinf(loss).any(), "Loss should not be Inf"
        assert loss.item() >= 0.0, "Loss should be non-negative"
        
        print("✓ Forward pass produces valid scalar loss")
    
    def test_masking_density_matches_intent(self):
        """Verify that higher masking density traces get higher cluster weighting."""
        B, C, D, H, W = 1, 1, 32, 8, 8
        reference = torch.randn(B, C, D, H, W)
        
        # Create mask with clear gradient: top row has no masking, bottom row is heavily masked
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :, 0, :] = 1.0    # Top row: 100% valid (0% masked)
        valid_mask[:, :, :, -1, :] = 0.1   # Bottom row: 10% valid (90% masked)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_weight_map, cluster_weight_map = criterion.diagnostic_weight_maps(reference, valid_mask)
        
        # Top row (low density) should have higher base weights
        top_base = base_weight_map[0, 0, 0, 0, :].mean().item()
        # Bottom row (high density) should have higher cluster weights
        bottom_cluster = cluster_weight_map[0, 0, 0, -1, :].mean().item()
        
        assert top_base > 0.1, f"Top row should have significant base weighting, got {top_base}"
        assert bottom_cluster > 0.1, f"Bottom row should have significant cluster weighting, got {bottom_cluster}"
        
        print("✓ Masking density correctly drives base vs. cluster weighting")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
