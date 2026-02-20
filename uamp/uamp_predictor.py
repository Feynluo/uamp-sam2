# -*- coding: utf-8 -*-
import torch
import numpy as np
from samurai import SamuraiPredictor
from uamp.memory_controller import UAMPMemoryController

class UAMPSamuraiPredictor(SamuraiPredictor):
    """
    UAMP predictor based on SAMURAI
    Fully preserves the SAMURAI native interface, replacing only the memory update logic with UAMP
    """
    def __init__(self, sam2_model):
        super().__init__(sam2_model)
        # Initialize UAMP memory controller (adapt to SAM2 feature dim)
        self.feat_dim = sam2_model.image_encoder.out_dim
        self.uamp_mem_ctrl = UAMPMemoryController(feat_dim=self.feat_dim).to(self.device)
        # UAMP state cache (aligned with SAMURAI video state)
        self.uamp_prev_img = None
        self.uamp_prev_feat = None
        self.uamp_feat_bank = []  # UAMP historical feature bank

    def reset_video(self):
        """Reset video state (compatible with SAMURAI native call)"""
        super().reset_video()
        self.uamp_prev_img = None
        self.uamp_prev_feat = None
        self.uamp_feat_bank = []

    def _preprocess_img(self, img):
        """Image preprocessing (aligned with SAM2/SAMURAI/MatAnyone format)"""
        if isinstance(img, np.ndarray):
            # Convert to tensor + normalize + resize (consistent with SAM2 input)
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            img = img.unsqueeze(0).to(self.device)
        return img

    def predict_video_frame(self, frame, **kwargs):
        """
        Override SAMURAI's video frame prediction to inject UAMP logic
        frame: raw video frame (H, W, 3) np.uint8
        return: masks, scores, logits (identical to SAMURAI outputs)
        """
        # 1. SAMURAI native preprocessing: record size + image tensor conversion
        original_size = frame.shape[:2]
        frame_tensor = self._preprocess(frame).to(self.device)
        # UAMP image preprocessing (normalization, used for motion uncertainty)
        uamp_curr_img = self._preprocess_img(frame)

        # 2. extract SAM2 native image features
        with torch.no_grad():
            curr_feat = self.model.image_encoder(frame_tensor)  # (1,C,H,W)

        # 3. UAMP core logic: feature enhancement (no UAMP for first frame; use native features)
        if self.uamp_prev_feat is not None and len(self.uamp_feat_bank) > 0:
            curr_feat, _ = self.uamp_mem_ctrl(
                curr_img=uamp_curr_img,
                prev_img=self.uamp_prev_img,
                curr_feat=curr_feat,
                prev_feat=self.uamp_prev_feat,
                feat_bank=self.uamp_feat_bank
            )

        # 4. feed UAMP-enhanced features into SAM2 decoder (replace native feature cache)
        self.model._cached_image_embeddings = {
            'image_embedding': curr_feat,
            'image_pe': self.model.image_encoder.get_dense_pe().to(curr_feat.device)
        }

        # 5. SAMURAI native decoding and prediction (compatible with all kwargs)
        masks, scores, logits = self.predict(
            point_coords=None,
            point_labels=None,
            mask_input=self._prev_mask_logit,
            multimask_output=False,
            **kwargs
        )

        # 6. update UAMP+SAMURAI state
        self._prev_mask_logit = logits
        self.uamp_prev_img = uamp_curr_img
        self.uamp_prev_feat = curr_feat.detach()
        self.uamp_feat_bank.append(curr_feat.detach())
        # memory bank pruning (avoid OOM, align with MatAnyone memory sparsity strategy)
        if len(self.uamp_feat_bank) > 50:
            self.uamp_feat_bank = self.uamp_feat_bank[::2]

        # 7. post-processing (identical to SAMURAI output format)
        masks = masks[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        logits = logits[0].cpu().numpy()

        return masks, scores, logits
