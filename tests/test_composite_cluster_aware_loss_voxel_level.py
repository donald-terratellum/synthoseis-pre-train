#!/usr/bin/env python3
"""Test cases for CompositeClusterAwareLoss with voxel-level weight assignment."""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss


class TestVoxelLevelClusterAwareLoss:
    """Test suite for voxel-level weight assignment in CompositeClusterAwareLoss."""
    
    def test_voxel_level_partition(self):
        """Verify that base and cluster masks partition all voxels at voxel level."""
        B, C, D, H, W = 1, 1, 32, 16, 16
        reference = torch.randn(B, C, D, H, W)
        
        # Create mask: some voxels valid, some masked
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :, :, :8] = 1.0   # Left half: all valid
        valid_mask[:, :, :, :, 8:] = 0.0   # Right half: all masked
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, valid_mask)
        
        # Check shapes
        assert base_mask.shape == (B, C, D, H, W)
        assert cluster_mask.shape == (B, C, D, H, W)
        
        # Base and cluster should partition all voxels
        partition = base_mask + cluster_mask
        assert torch.allclose(partition, torch.ones_like(partition)), \
            "Base + cluster masks should cover all voxels"
        
        # Left half should be base, right half should be cluster
        left_base = base_mask[:, :, :, :, :8]
        right_cluster = cluster_mask[:, :, :, :, 8:]
        
        assert torch.allclose(left_base, torch.ones_like(left_base)), \
            "Valid voxels should be in base mask"
        assert torch.allclose(right_cluster, torch.ones_like(right_cluster)), \
            "Masked voxels should be in cluster mask"
        
        print("✓ Voxel-level partition correct")
    
    def test_weight_maps_show_valid_vs_masked_regions(self):
        """Verify diagnostic weight maps correctly show valid and masked regions."""
        B, C, D, H, W = 1, 1, 24, 12, 12
        reference = torch.randn(B, C, D, H, W)
        
        # Create mask: top half valid, bottom half masked
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, :6, :, :] = 1.0   # Top valid
        valid_mask[:, :, 6:, :, :] = 0.0   # Bottom masked
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss, base_weight=1.0/3.0, cluster_weight=2.0/3.0)
        
        base_weight_map, cluster_weight_map = criterion.diagnostic_weight_maps(reference, valid_mask)
        
        # Top half should have base weights
        top_base = base_weight_map[:, :, :6, :, :].mean().item()
        assert top_base > 0.1, f"Valid voxels should have base weight, got {top_base}"
        
        # Bottom half should have cluster weights
        bottom_cluster = cluster_weight_map[:, :, 6:, :, :].mean().item()
        assert bottom_cluster > 0.1, f"Masked voxels should have cluster weight, got {bottom_cluster}"
        
        # Top half cluster should be zero
        top_cluster = cluster_weight_map[:, :, :6, :, :].sum().item()
        assert top_cluster < 1e-6, f"Valid voxels should have no cluster weight, got {top_cluster}"
        
        # Bottom half base should be zero
        bottom_base = base_weight_map[:, :, 6:, :, :].sum().item()
        assert bottom_base < 1e-6, f"Masked voxels should have no base weight, got {bottom_base}"
        
        print("✓ Weight maps correctly distinguish valid and masked regions")
    
    def test_voxel_counts_partition_correctly(self):
        """Verify that voxel counts in base + cluster sum to total volume."""
        B, C, D, H, W = 1, 1, 30, 15, 15
        reference = torch.randn(B, C, D, H, W)
        
        # Create mask with ~60% valid, ~40% masked
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, 12:, :, :] = 0.0  # Mask bottom 40%
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_weight_map, cluster_weight_map = criterion.diagnostic_weight_maps(reference, valid_mask)
        
        # Count nonzero weights
        base_count = (base_weight_map > 1e-6).sum().item()
        cluster_count = (cluster_weight_map > 1e-6).sum().item()
        total_expected = D * H * W
        
        assert base_count + cluster_count == total_expected, \
            f"Base {base_count} + cluster {cluster_count} should equal total {total_expected}"
        
        # Base should be ~60%, cluster ~40% (approximately)
        base_pct = 100.0 * base_count / total_expected
        cluster_pct = 100.0 * cluster_count / total_expected
        
        assert 55 < base_pct < 65, f"Base should be ~60%, got {base_pct:.1f}%"
        assert 35 < cluster_pct < 45, f"Cluster should be ~40%, got {cluster_pct:.1f}%"
        
        print(f"✓ Voxel counts partition correctly: base {base_pct:.1f}%, cluster {cluster_pct:.1f}%")
    
    def test_no_mask_defaults_to_all_base(self):
        """Verify that omitting valid_mask results in all base weighting."""
        B, C, D, H, W = 1, 1, 16, 8, 8
        reference = torch.randn(B, C, D, H, W)
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        base_mask, cluster_mask = criterion._weight_masks(reference, None)
        
        # All voxels should be base
        assert torch.allclose(base_mask, torch.ones_like(base_mask))
        assert torch.allclose(cluster_mask, torch.zeros_like(cluster_mask))
        
        print("✓ No mask correctly defaults to all base weighting")
    
    def test_forward_pass_produces_valid_loss(self):
        """Verify forward pass produces valid scalar loss."""
        B, C, D, H, W = 2, 1, 32, 12, 12
        pred = torch.randn(B, C, D, H, W)
        target = torch.randn(B, C, D, H, W)
        
        # Create mask: half valid, half masked
        valid_mask = torch.ones(B, C, D, H, W)
        valid_mask[:, :, 16:, :, :] = 0.0
        
        base_loss = SSIMMSELoss3D(alpha=0.1)
        criterion = CompositeClusterAwareLoss(base_loss)
        
        loss = criterion(pred, target, valid_mask)
        
        # Should be scalar
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.item() >= 0.0
        
        print("✓ Forward pass produces valid scalar loss")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
