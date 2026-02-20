# UAMP (Uncertainty-Aware Memory Propagation) for SAM2
This repository implements the **UAMP method** based on SAMURAI (SAM2 extension) and MatAnyone's appearance uncertainty estimation logic, supporting end-to-end fine-tuning on DAVIS/YouTube-VOS datasets (freezing SAM2 backbone, only fine-tuning lightweight UAMP layers) for improved video object segmentation (VOS) performance.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Model Checkpoints](#3-model-checkpoints)
3. [Inference (Zero-Shot)](#4-inference-zero-shot)
4. [End-to-End Fine-Tuning](#5-end-to-end-fine-tuning)
5. [Citation](#6-citation)

---

## 1. Environment Setup
### Basic Dependencies
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
install `requirements.txt`.

## 2. Model Checkpoints
### Download SAM2 Pre-trained Weights
```bash
mkdir -p checkpoints
cd checkpoints
# Download SAM2-Hiera-L (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ..
```
## 3. Inference (Zero-Shot)
### 3.1 Single Video Inference
```bash
python demo_uamp.py \
    --sam2_checkpoint checkpoints/sam2_hiera_large.pt \
    --sam2_model_type sam2_hiera_l \
    --input_video path/to/your/video.mp4 \
    --output_video path/to/output/uamp_result.mp4 \
    --device cuda:0
```

### 3.2 Inference on DAVIS/YouTube-VOS
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

## 4. End-to-End Fine-Tuning
### 4.1 Fine-Tuning Configuration
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

### 4.2 Fine-Tuning Script (`train_uamp.py`)

### 4.3 Dataset Loader (`datasets/davis_dataset.py`)

### 4.4 Run Fine-Tuning
```bash
python train_uamp.py --config configs/finetune_uamp.yaml
```

---

## 5. Citation
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
