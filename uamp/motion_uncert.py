# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.flow as flow_ops

class MotionUncert(nn.Module):
    """
    Motion uncertainty estimation module (compatible with MatAnyone appearance branch)
    Core: flow magnitude (motion speed) + flow gradient (motion smoothness); higher speed/gradient => higher motion uncertainty
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.flow_proj = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        self.fusion = nn.Conv2d(2, 1, 3, 1, 1, bias=False)

    @torch.no_grad()
    def compute_flow(self, img_prev, img_curr):
        """
        Compute inter-frame optical flow (aligned to SAMURAI/MatAnyone image input format)
        img_prev/img_curr: normalized images (B,3,H,W) [0,1]
        """
        # Convert to grayscale (color not needed for flow computation)
        img_prev_gray = kornia.color.rgb_to_grayscale(img_prev)
        img_curr_gray = kornia.color.rgb_to_grayscale(img_curr)
        # RAFT optical flow (lightweight, matches MatAnyone compute efficiency)
        flow = flow_ops.raft_flow(img_prev_gray, img_curr_gray, num_iterations=6)
        return flow

    def forward(self, img_prev, img_curr):
        """
        Args:
            img_prev/img_curr: previous/current normalized images (B,3,H,W)
        Returns:
            motion_uncert: motion uncertainty mask (B,1,H,W) values in [0,1]
        """
        # Step 1: compute inter-frame optical flow
        flow = self.compute_flow(img_prev, img_curr)  # (B,2,H,W)
        # Step 2: flow magnitude (motion speed)
        flow_mag = torch.norm(flow, dim=1, keepdim=True)
        flow_mag = F.normalize(flow_mag, dim=[2,3], p=2)
        # Step 3: flow gradient (motion smoothness; large gradient indicates motion discontinuity)
        flow_grad = torch.abs(F.avg_pool2d(flow, 3, 1, 1) - flow)
        flow_grad_mag = torch.norm(flow_grad, dim=1, keepdim=True)
        flow_grad_mag = F.normalize(flow_grad_mag, dim=[2,3], p=2)
        # Step 4: fuse motion cues (align output range with MatAnyone appearance branch)
        fusion_feat = torch.cat([flow_mag, flow_grad_mag], dim=1)
        motion_uncert = torch.sigmoid(self.fusion(fusion_feat))

        return motion_uncert
