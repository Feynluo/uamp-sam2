# UAMP (Uncertainty-Aware Memory Propagation) for SAM2
This repository implements the **UAMP method** based on SAMURAI (SAM2 extension) and MatAnyone's appearance uncertainty estimation logic, supporting end-to-end fine-tuning on DAVIS/YouTube-VOS datasets (freezing SAM2 backbone, only fine-tuning lightweight UAMP layers) for improved video object segmentation (VOS) performance.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Model Checkpoints](#3-model-checkpoints)
4. [Inference (Zero-Shot)](#4-inference-zero-shot)
5. [End-to-End Fine-Tuning](#5-end-to-end-fine-tuning)
6. [Citation](#6-citation)

---

## 1. Environment Setup
### 1.1 Basic Dependencies
```bash
# Clone repository
git clone https://github.com/your-username/uamp-sam2.git
cd uamp-sam2

# Create conda environment (recommended)
conda create -n uamp-sam2 python=3.10
conda activate uamp-sam2

# Install PyTorch (match CUDA version)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Install SAM2 (from SAMURAI)
cd sam2 && pip install -e . && pip install -e ".[notebooks]" && cd ..
```

### 1.2 Requirements.txt
Create `requirements.txt` with the following content:
```txt
matplotlib==3.7.5
tikzplotlib==0.10.1
jpeg4py==0.1.4
opencv-python==4.9.0.80
lmdb==1.4.1
pandas==2.1.4
scipy==1.11.4
loguru==0.7.2
kornia==0.7.2
einops==0.7.0
timm==0.9.16
ffmpeg-python==0.2.0
gradio==4.13.0
pillow==10.1.0
pycocotools==2.0.7
albumentations==1.4.1
wandb==0.16.1
```

---

## 2. Dataset Preparation
### 2.1 DAVIS 2017
```bash
# Create dataset directory
mkdir -p datasets/DAVIS

# Download DAVIS 2017 (480p)
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -P datasets/DAVIS
unzip datasets/DAVIS/DAVIS-2017-trainval-480p.zip -d datasets/DAVIS

# Organize structure
mv datasets/DAVIS/DAVIS/Annotations_480p datasets/DAVIS/annotations
mv datasets/DAVIS/DAVIS/JPEGImages/480p datasets/DAVIS/images
rm -rf datasets/DAVIS/DAVIS
```

### 2.2 YouTube-VOS 2019
```bash
# Create dataset directory
mkdir -p datasets/YouTubeVOS

# Download YouTube-VOS 2019 (train/val) from official website: https://youtube-vos.org/dataset/
# After download, organize as follows:
datasets/YouTubeVOS/
├── train/
│   ├── JPEGImages/
│   └── annotations/
└── val/
    ├── JPEGImages/
    └── annotations/
```

### 2.3 Dataset Structure
Final dataset structure:
```
datasets/
├── DAVIS/
│   ├── images/        # Frame images (480p)
│   └── annotations/   # Segmentation masks
└── YouTubeVOS/
    ├── train/
    │   ├── JPEGImages/
    │   └── annotations/
    └── val/
        ├── JPEGImages/
        └── annotations/
```

---

## 3. Model Checkpoints
### 3.1 Download SAM2 Pre-trained Weights
```bash
mkdir -p checkpoints
cd checkpoints
# Download SAM2-Hiera-L (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ..
```

### 3.2 Pre-trained UAMP Weights (Optional)
Download fine-tuned UAMP weights (after training) to `checkpoints/uamp_finetuned.pth`.

---

## 4. Inference (Zero-Shot)
### 4.1 Single Video Inference
```bash
python demo_uamp.py \
    --sam2_checkpoint checkpoints/sam2_hiera_large.pt \
    --sam2_model_type sam2_hiera_l \
    --input_video path/to/your/video.mp4 \
    --output_video path/to/output/uamp_result.mp4 \
    --device cuda:0
```

### 4.2 Inference on DAVIS/YouTube-VOS
```bash
python eval_uamp.py \
    --dataset DAVIS \
    --dataset_root datasets/DAVIS \
    --sam2_checkpoint checkpoints/sam2_hiera_large.pt \
    --uamp_checkpoint checkpoints/uamp_finetuned.pth \
    --device cuda:0 \
    --save_results
```

---

## 5. End-to-End Fine-Tuning
### 5.1 Fine-Tuning Configuration
Create `configs/finetune_uamp.yaml`:
```yaml
# General settings
experiment_name: uamp_finetune_davis
device: cuda:0
seed: 42
num_workers: 8
save_dir: checkpoints/finetune

# Dataset settings
dataset: DAVIS
dataset_root: datasets/DAVIS
batch_size: 2
num_frames: 8  # Number of frames per video clip
image_size: 480  # Match DAVIS 480p

# Model settings
sam2_checkpoint: checkpoints/sam2_hiera_large.pt
sam2_model_type: sam2_hiera_l
freeze_sam2_backbone: true  # Freeze SAM2 encoder/decoder
freeze_sam2_image_encoder: true
freeze_sam2_mask_decoder: true

# Optimization settings
lr: 1e-4  # Only for UAMP layers
weight_decay: 1e-5
num_epochs: 50
warmup_epochs: 5
lr_scheduler: cosine  # Cosine annealing
optimizer: adamw

# Logging
log_interval: 10  # Log every 10 batches
val_interval: 5   # Validate every 5 epochs
use_wandb: true   # Optional: for experiment tracking
```

### 5.2 Fine-Tuning Script (`train_uamp.py`)
```python
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

# Import UAMP and dataset modules
from uamp.uamp_predictor import UAMPSamuraiPredictor
from uamp.memory_controller import UAMPMemoryController
from datasets.davis_dataset import DAVISDataset
from sam2.build_sam import build_sam2
from sam2.sam2_model_registry import sam2_model_registry

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Freeze SAM2 backbone
def freeze_sam2(model, freeze_encoder=True, freeze_decoder=True):
    """Freeze SAM2 backbone layers (image encoder/mask decoder)"""
    if freeze_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        logger.info("SAM2 image encoder frozen")
    
    if freeze_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = False
        logger.info("SAM2 mask decoder frozen")
    return model

# Get UAMP trainable parameters
def get_uamp_trainable_params(uamp_ctrl):
    """Get only UAMP lightweight layers (projection/fusion) for fine-tuning"""
    trainable_params = []
    for name, param in uamp_ctrl.named_parameters():
        if "feat_proj" in name or "fusion" in name or "attn_fusion" in name:
            param.requires_grad = True
            trainable_params.append(param)
            logger.info(f"UAMP trainable layer: {name}")
        else:
            param.requires_grad = False
    return trainable_params

# Validation loop
def validate(model, uamp_ctrl, val_loader, criterion, device):
    model.eval()
    uamp_ctrl.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            frames, masks, video_names = batch
            frames = frames.to(device)
            masks = masks.to(device)
            
            # Forward pass
            pred_masks = []
            for i in range(frames.shape[1]):  # Iterate over frames
                frame = frames[:, i]
                mask_pred, _, _ = model.predict_video_frame(frame)
                pred_masks.append(torch.from_numpy(mask_pred).to(device))
            
            pred_masks = torch.stack(pred_masks, dim=1)
            loss = criterion(pred_masks, masks)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# Main training loop
def main(config):
    # Set seed
    torch.manual_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize SAM2 model
    sam2_model = sam2_model_registry[config['sam2_model_type']](
        checkpoint=config['sam2_checkpoint']
    )
    sam2_model = freeze_sam2(
        sam2_model,
        freeze_encoder=config['freeze_sam2_image_encoder'],
        freeze_decoder=config['freeze_sam2_mask_decoder']
    )
    sam2_model = sam2_model.to(config['device'])
    
    # Initialize UAMP controller
    uamp_ctrl = UAMPMemoryController(feat_dim=sam2_model.image_encoder.out_dim)
    uamp_ctrl = uamp_ctrl.to(config['device'])
    
    # Get trainable parameters (only UAMP lightweight layers)
    trainable_params = get_uamp_trainable_params(uamp_ctrl)
    
    # Initialize dataset and dataloader
    train_dataset = DAVISDataset(
        root=config['dataset_root'],
        split='train',
        num_frames=config['num_frames'],
        image_size=config['image_size']
    )
    val_dataset = DAVISDataset(
        root=config['dataset_root'],
        split='val',
        num_frames=config['num_frames'],
        image_size=config['image_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - config['warmup_epochs']
    )
    criterion = nn.BCEWithLogitsLoss()  # Segmentation loss
    
    # Initialize UAMP predictor
    predictor = UAMPSamuraiPredictor(sam2_model)
    predictor.uamp_mem_ctrl = uamp_ctrl
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        predictor.model.train()
        uamp_ctrl.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            frames, masks, video_names = batch
            frames = frames.to(config['device'])
            masks = masks.to(config['device'])
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (video clip)
            pred_masks = []
            for i in range(frames.shape[1]):
                frame = frames[:, i]
                mask_pred, _, _ = predictor.predict_video_frame(frame)
                pred_masks.append(torch.from_numpy(mask_pred).to(config['device']))
            
            pred_masks = torch.stack(pred_masks, dim=1)
            loss = criterion(pred_masks, masks)
            
            # Backward pass (only UAMP layers)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Log training progress
            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        if epoch >= config['warmup_epochs']:
            scheduler.step()
        
        # Validation
        if (epoch + 1) % config['val_interval'] == 0:
            val_loss = validate(predictor.model, uamp_ctrl, val_loader, criterion, config['device'])
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config['save_dir'], f"uamp_best_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'uamp_state_dict': uamp_ctrl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, save_path)
                logger.info(f"Best model saved to {save_path}")
    
    # Save final model
    final_save_path = os.path.join(config['save_dir'], "uamp_finetuned_final.pth")
    torch.save(uamp_ctrl.state_dict(), final_save_path)
    logger.info(f"Final UAMP model saved to {final_save_path}")

if __name__ == "__main__":
    config = load_config("configs/finetune_uamp.yaml")
    logger.info(f"Starting UAMP fine-tuning with config: {config['experiment_name']}")
    main(config)
```

### 5.3 Dataset Loader (`datasets/davis_dataset.py`)
```python
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
```

### 5.4 Run Fine-Tuning
```bash
python train_uamp.py --config configs/finetune_uamp.yaml
```

---

## 6. Citation
If you use this code, please cite the relevant works:
```bibtex
@misc{sam2,
    title={Segment Anything 2},
    author={Meta AI},
    year={2024},
    url={https://github.com/facebookresearch/sam2}
}
@misc{samurai,
    title={SAMURAI: SAM2 for Video Segmentation},
    author={Yang, Chris},
    year={2024},
    url={https://github.com/yangchris11/samurai}
}
@misc{matanyone,
    title={MatAnyone: Universal Image Matting with SAM},
    author={Yang, PQ and others},
    year={2024},
    url={https://github.com/pq-yang/MatAnyone}
}
```
