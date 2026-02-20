
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DAVISDataset(Dataset):
    """DAVIS 2017 Dataset for UAMP fine-tuning"""
    def __init__(self, root, split='train', num_frames=8, image_size=480):
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        
        # Dataset paths
        self.images_dir = os.path.join(root, "images")
        self.annotations_dir = os.path.join(root, "annotations")
        
        # Get video list
        self.video_list = self._get_video_list()
    
    def _get_video_list(self):
        """Get list of videos for train/val split"""
        split_file = os.path.join(self.root, "ImageSets/2017", f"{self.split}.txt")
        with open(split_file, 'r') as f:
            video_list = [line.strip() for line in f.readlines()]
        return video_list
    
    def _load_frames(self, video_name):
        """Load video frames and resize"""
        frame_dir = os.path.join(self.images_dir, video_name)
        frame_files = sorted(os.listdir(frame_dir))
        
        # Sample consecutive frames
        start_idx = np.random.randint(0, max(1, len(frame_files) - self.num_frames + 1))
        frame_files = frame_files[start_idx:start_idx + self.num_frames]
        
        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(frame_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            frames.append(img)
        
        return np.array(frames)
    
    def _load_masks(self, video_name):
        """Load segmentation masks and resize"""
        mask_dir = os.path.join(self.annotations_dir, video_name)
        mask_files = sorted(os.listdir(mask_dir))
        
        # Sample matching frames
        start_idx = np.random.randint(0, max(1, len(mask_files) - self.num_frames + 1))
        mask_files = mask_files[start_idx:start_idx + self.num_frames]
        
        masks = []
        for f in mask_files:
            mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 0).astype(np.float32)  # Binary mask
            masks.append(mask)
        
        return np.array(masks)
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        
        # Load frames and masks
        frames = self._load_frames(video_name)
        masks = self._load_masks(video_name)
        
        # Convert to torch tensors
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        masks = torch.from_numpy(masks).unsqueeze(1).float()  # (T, 1, H, W)
        
        return frames, masks, video_name
