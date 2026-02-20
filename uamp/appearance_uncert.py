
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MatAnyoneStyleAppearanceUncert(nn.Module):
    """
    MatAnyone-style appearance uncertainty estimation module.
    Core logic:
    1. Pixel-wise L2 distance between frames.
    2. Propagation discrepancy between memory features and current features.
    3. Feature variance sparsity.
    Output: appearance_uncert (0~1, higher values indicate less reliable appearance due to occlusion, blur, or deformation)
    """
    def __init__(self, feat_dim=256, hidden_dim=64):
        super().__init__()
        # Align with MatAnyone's feature projection design (lightweight conv + BN, no activation to avoid feature distortion)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        self.fusion = nn.Conv2d(3, 1, 3, 1, 1, bias=False)
        self.eps = 1e-6

    def forward(self, curr_feat, prev_feat, mem_feat):
        """
        Args:
            curr_feat: current-frame SAM2 image features (B, C, H, W)
            prev_feat: previous-frame SAM2 image features (B, C, H, W)
            mem_feat: historical memory features (MatAnyone-style, global average pooling) (B, C, H, W)
        Returns:
            appearance_uncert: appearance uncertainty mask (B, 1, H, W) values in [0,1]
        """
        B, C, H, W = curr_feat.shape
        # Step 1: feature projection (align MatAnyone feature dim, reduce computation)
        curr_feat_proj = self.feat_proj(curr_feat)
        prev_feat_proj = self.feat_proj(prev_feat)
        mem_feat_proj = self.feat_proj(mem_feat)

        # Step 2: MatAnyone core 1 - inter-frame appearance feature L2 distance
        frame_feat_diff = torch.norm(curr_feat_proj - prev_feat_proj, dim=1, keepdim=True)
        frame_feat_diff = F.normalize(frame_feat_diff, dim=[2,3], p=2)

        # Step 3: MatAnyone core 2 - memory-to-current feature propagation discrepancy
        mem_feat_diff = torch.norm(curr_feat_proj - mem_feat_proj, dim=1, keepdim=True)
        mem_feat_diff = F.normalize(mem_feat_diff, dim=[2,3], p=2)

        # Step 4: MatAnyone core 3 - feature variance sparsity (smaller variance => more uniform features => higher uncertainty)
        feat_var = torch.var(curr_feat_proj, dim=1, keepdim=True) + self.eps
        feat_sparsity = 1.0 - F.normalize(feat_var, dim=[2,3], p=2)

        # Step 5: fuse three appearance-uncertainty cues (MatAnyone-style direct fusion)
        fusion_feat = torch.cat([frame_feat_diff, mem_feat_diff, feat_sparsity], dim=1)
        appearance_uncert = torch.sigmoid(self.fusion(fusion_feat))

        return appearance_uncert

    @staticmethod
    def matanyone_style_mem_feat(feat_bank):
        """
        Generate MatAnyone-style historical memory feature
        feat_bank: list of historical features [(B, C, H, W), ...]
        return: global-average-pooled memory feature (B, C, H, W)
        """
        feat_stack = torch.stack(feat_bank, dim=2)  # (B, C, N, H, W)
        mem_feat = torch.mean(feat_stack, dim=2)    # align with MatAnyone memory averaging strategy
        return mem_feat