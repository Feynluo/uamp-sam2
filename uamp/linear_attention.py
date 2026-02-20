
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LinearAttention(nn.Module):
    """
    Linear attention module (O(N) complexity, aligned with SAM2/MatAnyone feature dims)
    Supports short-term/long-term memory queries, suitable for long video sequences
    """
    def __init__(self, feat_dim=256, temp_scale=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temp_scale = temp_scale
        # Lightweight projection layers (no BN to avoid disturbing temporal attention features)
        self.q_proj = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)

    def forward(self, query, key, value, memory_type="short-term"):
        """
        Args:
            query: query features from current frame (B, C, H, W)
            key: key features from memory bank (B, C, N, H, W), N = number of memory frames
            value: value features from memory bank (B, C, N, H, W)
            memory_type: short-term/long-term attention type
        Returns:
            attn_feat: attention-weighted features from linear attention (B, C, H, W)
        """
        B, C, N, H, W = key.shape
        # Feature projections
        q = self.q_proj(query).view(B, C, -1)  # (B,C,HW)
        k = self.k_proj(key).view(B, C, N*H*W) # (B,C,NHW)
        v = self.v_proj(value).view(B, C, N*H*W)# (B,C,NHW)

        # Short/long-term attention masks (aligned with UAMP memory strategy)
        if memory_type == "short-term":
            # Short-term: focus only on the most recent 5 frames, mask earlier frames
            stm_mask = torch.zeros(B, 1, N*H*W, device=query.device)
            stm_mask[:, :, -5*H*W:] = 1.0
            k = k * stm_mask
            v = v * stm_mask
        elif memory_type == "long-term":
            # Long-term: global uniform sampling to avoid redundant computation (align with MatAnyone memory sparsity update)
            ltm_idx = torch.linspace(0, N*H*W-1, steps=H*W, device=query.device).long()
            k = k[:, :, ltm_idx]
            v = v[:, :, ltm_idx]

        # Linear attention computation (no full matrix multiply, low complexity)
        q = F.softmax(q / self.temp_scale, dim=-1)
        k = F.softmax(k / self.temp_scale, dim=1)
        attn = torch.bmm(q.transpose(1,2), k)
        attn_feat = torch.bmm(attn, v.transpose(1,2)).transpose(1,2)
        # Reshape back to original feature size
        attn_feat = attn_feat.view(B, C, H, W)

        return attn_feat
