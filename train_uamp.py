
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
