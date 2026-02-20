
import torch
import torch.nn as nn
import torch.nn.functional as F
from uamp.appearance_uncert import MatAnyoneStyleAppearanceUncert
from uamp.motion_uncert import MotionUncert
from uamp.linear_attention import LinearAttention

class UAMPMemoryController(nn.Module):
    """
    UAMP memory controller (core replacement module for the SAMURAI main framework)
    Pipeline: dual-uncertainty estimation -> uncertainty mask fusion -> short/long-term linear attention -> adaptive feature output
    """
    def __init__(self, feat_dim=256, short_term_len=5):
        super().__init__()
        self.short_term_len = short_term_len
        # 1. Uncertainty modules (appearance = MatAnyone-style, motion = compatible paradigm)
        self.app_uncert = MatAnyoneStyleAppearanceUncert(feat_dim=feat_dim)
        self.mot_uncert = MotionUncert()
        # 2. Uncertainty fusion gate (aligned with MatAnyone's Uncertainty Head design)
        self.uncert_fusion = nn.Conv2d(2, 1, 1, bias=False)
        # 3. Short/long-term linear attention (core memory update)
        self.stm_attn = LinearAttention(feat_dim=feat_dim)  # short-term memory
        self.ltm_attn = LinearAttention(feat_dim=feat_dim)  # long-term memory
        # 4. Attention fusion gate (guided by the uncertainty mask)
        self.attn_fusion = nn.Conv2d(2*feat_dim, feat_dim, 1, bias=False)

    def forward(self, curr_img, prev_img, curr_feat, prev_feat, feat_bank):
        """
        Args:
            curr_img/prev_img: current/previous normalized images (B,3,H,W)
            curr_feat/prev_feat: current/previous SAM2 features (B,C,H,W)
            feat_bank: SAMURAI historical feature bank [(B,C,H,W), ...]
        Returns:
            uamp_feat: UAMP-enhanced features (fed into SAM2 decoder) (B,C,H,W)
            uncert_mask: fused uncertainty mask (B,1,H,W) values in [0,1]
        """
        B, C, H, W = curr_feat.shape
        # Step 1: generate MatAnyone-style historical memory feature
        mem_feat = self.app_uncert.matanyone_style_mem_feat(feat_bank)
        # Step 2: dual uncertainty estimation
        app_uncert = self.app_uncert(curr_feat, prev_feat, mem_feat)
        mot_uncert = self.mot_uncert(prev_img, curr_img)
        # Step 3: fuse uncertainty masks (MatAnyone sigmoid fusion style)
        uncert_fusion = torch.cat([app_uncert, mot_uncert], dim=1)
        uncert_mask = torch.sigmoid(self.uncert_fusion(uncert_fusion))

        # Step 4: construct short/long-term memory bank (align to linear attention input format)
        feat_stack = torch.stack(feat_bank, dim=2) if feat_bank else curr_feat.unsqueeze(2)
        # Step 5: compute short/long-term linear attention
        stm_feat = self.stm_attn(curr_feat, feat_stack, feat_stack, memory_type="short-term")
        ltm_feat = self.ltm_attn(curr_feat, feat_stack, feat_stack, memory_type="long-term")

        # Step 6: uncertainty-guided attention fusion (core idea)
        # High uncertainty (occlusion/deformation/fast motion) -> rely on long-term memory (LTM, stable)
        # Low uncertainty -> rely on short-term memory (STM, detail)
        stm_weight = 1.0 - uncert_mask
        ltm_weight = uncert_mask
        fused_attn_feat = torch.cat([stm_feat * stm_weight, ltm_feat * ltm_weight], dim=1)
        uamp_feat = self.attn_fusion(fused_attn_feat)

        return uamp_feat, uncert_mask
