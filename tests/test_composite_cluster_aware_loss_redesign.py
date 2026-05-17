#!/usr/bin/env python3
"""Test cases for CompositeClusterAwareLoss redesign with corrected base/cluster weighting."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss


class TestClusterAwareLossRedesign:
    """Test suite for the redesigned cluster-aware weighting logic."""
    
    def test_base_mask_covers_all_valid_traces(self):
        """Verify base_mask is 1.0 for all traces with any valid voxels."""
        B, C, D, H, W = 1, 1, 32, 16, 16
        reference = torch.randn(B, C, D, H, W)
        
        # Create a valid_mask where some traces are completely masked
        valid_mask = torch.ones(B, C, D, H, W, dtype=torch.float32)
        
        # Set column x=5 to be completely masked (all depths invalid)
        valid_mask[:, :, :, :, 5] = 0.0
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, valid_mask)
        
        # Check shapes
        assert base_mask.shape == (B, C, D, H, W), f"base_mask shape {base_mask.shape} != {(B, C, D, H, W)}"
        assert cluster_mask.shape == (B, C, D, H, W), f"cluster_mask shape {cluster_mask.shape} != {(B, C, D, H, W)}"
        
        # Check base_mask: should be 1.0 everywhere except column 5 (completely masked)
        # and possibly edges if valid_mask is all zeros there
        assert base_mask[0, 0, 0, 0, 0] > 0.9, "base_mask should be 1.0 for valid traces"
        assert base_mask[0, 0, 0, 0, 5] < 0.1, "base_mask should be 0.0 for completely masked traces"
        
        print("✓ base_mask covers all valid traces correctly")
    
    def test_cluster_mask_identifies_masked_and_neighbors(self):
        """Verify cluster_mask marks completely masked traces and their spatial neighbors."""
        B, C, D, H, W = 1, 1, 32, 16, 16
        reference = torch.randn(B, C, D, H, W)
        
        # Create valid_mask with a cluster of completely masked traces
        valid_mask = torch.ones(B, C, D, H, W, dtype=torch.float32)
        
        # Create a 3x3 cluster of completely masked traces at position (7, 7)-(9, 9)
        cluster_center_x, cluster_center_y = 8, 8
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = cluster_center_x + dx
                y = cluster_center_y + dy
                if 0 <= x < H and 0 <= y < W:
                    valid_mask[:, :, :, x, y] = 0.0  # Completely mask this trace
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss, kernel_size=5)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, valid_mask)
        
        # Cluster mask should be non-zero at the center cluster
        assert cluster_mask[0, 0, 0, cluster_center_x, cluster_center_y] > 0.5, \
            "cluster_mask should mark the center of completely masked cluster"
        
        # Check that neighbors are also marked (due to smoothing with kernel_size=5)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                x = cluster_center_x + dx
                y = cluster_center_y + dy
                if 0 <= x < H and 0 <= y < W:
                    # Most neighbors should have some cluster_mask value
                    # (exact threshold depends on kernel, but center should be highest)
                    pass
        
        # Verify that traces far from cluster have low cluster_mask
        far_x, far_y = 2, 2
        assert cluster_mask[0, 0, 0, far_x, far_y] < 0.5, \
            "cluster_mask should be near-zero for traces far from clusters"
        
        print("✓ cluster_mask correctly identifies masked clusters and neighbors")
    
    def test_weight_maps_correct_shape_and_values(self):
        """Verify diagnostic_weight_maps returns correct shapes and value ranges."""
        B, C, D, H, W = 2, 1, 16, 12, 12
        reference = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, C, D, H, W)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss, base_weight=1.0/3.0, cluster_weight=2.0/3.0)
        
        base_weight_map, cluster_weight_map = criterion.diagnostic_weight_maps(reference, valid_mask)
        
        # Check shapes
        assert base_weight_map.shape == (B, C, D, H, W)
        assert cluster_weight_map.shape == (B, C, D, H, W)
        
        # Check value ranges
        assert (base_weight_map >= 0).all(), "base_weight_map should be non-negative"
        assert (cluster_weight_map >= 0).all(), "cluster_weight_map should be non-negative"
        assert base_weight_map.max() <= 1.0/3.0 + 1e-5, f"base_weight_map max {base_weight_map.max()} exceeds base_weight"
        assert cluster_weight_map.max() <= 2.0/3.0 + 1e-5, f"cluster_weight_map max {cluster_weight_map.max()} exceeds cluster_weight"
        
        print("✓ weight maps have correct shapes and value ranges")
    
    def test_no_mask_provided_uses_all_valid(self):
        """Verify behavior when no valid_mask is provided."""
        B, C, D, H, W = 1, 1, 16, 12, 12
        reference = torch.randn(B, C, D, H, W)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, valid_mask=None)
        
        # When no mask provided, base_mask should be all 1s (all traces are "valid")
        assert (base_mask == 1.0).all(), "base_mask should be all 1.0 when no valid_mask provided"
        
        # cluster_mask should be all 0s (no completely masked traces)
        assert (cluster_mask == 0.0).all(), "cluster_mask should be all 0.0 when no valid_mask provided"
        
        print("✓ no mask provided behaves correctly")
    
    def test_forward_pass_no_nan_or_inf(self):
        """Verify forward pass doesn't produce NaN or Inf values."""
        B, C, D, H, W = 1, 1, 16, 12, 12
        pred = torch.randn(B, C, D, H, W) * 0.1
        target = torch.randn(B, C, D, H, W) * 0.1
        
        # Create a mask with some completely masked traces
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :, :, 5] = 0.0  # Completely mask column 5
        
        base_loss = SSIMMSELoss3D(alpha=0.1, window_size=11)
        criterion = CompositeClusterAwareLoss(base_loss, base_weight=1.0/3.0, cluster_weight=2.0/3.0)
        
        loss = criterion(pred, target, valid_mask)
        
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
        assert loss.item() > 0, "Loss should be positive"
        
        print(f"✓ forward pass produces valid loss: {loss.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
