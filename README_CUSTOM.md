# TransUNet for Chest X-ray Lung Segmentation

Custom adaptation of TransUNet for 2D Chest X-ray lung segmentation.

## Changes Made

### 1. **Legacy Compatibility Fixes**
- Fixed deprecated `scipy.ndimage.interpolation.zoom` import (now using `scipy.ndimage.zoom`)
- Updated [requirements.txt](requirements.txt) for PyTorch 2.x and numpy>=2.0 compatibility
- All code is now compatible with modern PyTorch and numpy versions

### 2. **Custom Dataset Loader**
- Created [datasets/dataset_custom.py](datasets/dataset_custom.py)
- Loads images from `CXR_png/` and masks from `masks/` directories
- Converts 1-channel grayscale X-rays to 3-channel RGB [3, 512, 512]
- Automatically resizes to 512x512 (configurable)
- Includes data augmentation (rotation, flipping)
- Auto-splits dataset into train/val (default 80/20 ratio)

### 3. **Training Script**
- Created [train_custom.py](train_custom.py) with comprehensive features:
  - **Argparse configuration**: All hyperparameters are configurable via command line
  - **AMP support**: Use `--use_amp` flag for Automatic Mixed Precision training
  - **Validation loop**: Calculates Dice Score after each epoch
  - **Best model saving**: Automatically saves best model based on validation Dice score
  - **TensorBoard logging**: Visualizes training progress, losses, and predictions
  - **Flexible configuration**: Easy to test locally with small batches or train on server

## Dataset Structure

Your dataset should be organized as:
```
data/Chest Xray Masks and Labels/data/Lung Segmentation/
├── CXR_png/
│   ├── CHNCXR_0001_0.png
│   ├── CHNCXR_0002_0.png
│   └── ...
└── masks/
    ├── CHNCXR_0001_0_mask.png
    ├── CHNCXR_0002_0_mask.png
    └── ...
```

## Installation

1. **Install dependencies:**
```bash
cd TransUNet
pip install -r requirements.txt
```

2. **Download pretrained weights:**
Download the R50-ViT-B_16 pretrained weights and place them at:
```
../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

You can download from the original TransUNet repository or train from scratch.

## Usage

### Quick Test (Local Development)
```bash
python train_custom.py \
    --batch_size 2 \
    --max_epochs 10 \
    --num_workers 2 \
    --img_size 512 \
    --base_lr 0.01
```

### Full Training with AMP (Server/GPU)
```bash
python train_custom.py \
    --batch_size 16 \
    --max_epochs 150 \
    --num_workers 8 \
    --img_size 512 \
    --base_lr 0.01 \
    --use_amp \
    --n_gpu 1
```

### All Available Arguments

**Data Parameters:**
- `--root_path`: Root directory containing CXR_png and masks subdirectories
- `--dataset`: Dataset name (default: ChestXray)
- `--num_classes`: Number of classes (default: 2 for binary segmentation)
- `--train_ratio`: Train/val split ratio (default: 0.8)

**Model Parameters:**
- `--vit_name`: ViT model variant (default: R50-ViT-B_16)
- `--n_skip`: Number of skip connections (default: 3)
- `--vit_patches_size`: ViT patch size (default: 16)
- `--img_size`: Input image size (default: 512)

**Training Parameters:**
- `--max_epochs`: Maximum number of epochs (default: 150)
- `--batch_size`: Batch size per GPU (default: 8)
- `--base_lr`: Base learning rate (default: 0.01)
- `--n_gpu`: Number of GPUs (default: 1)
- `--num_workers`: Number of data loading workers (default: 4)
- `--use_amp`: Enable Automatic Mixed Precision training
- `--save_interval`: Save checkpoint every N epochs (default: 10)
- `--seed`: Random seed (default: 1234)
- `--deterministic`: Use deterministic training (default: 1)
- `--output_dir`: Output directory for models (default: ../model)

## Training Output

The training script will create a directory structure:
```
../model/TU_ChestXray_512_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp/
├── log.txt              # Training logs
├── log/                 # TensorBoard logs
├── best_model.pth       # Best model (highest validation Dice)
├── epoch_9.pth          # Checkpoint at epoch 9
├── epoch_19.pth         # Checkpoint at epoch 19
└── ...
```

## Monitoring Training

Use TensorBoard to monitor training:
```bash
tensorboard --logdir ../model/TU_ChestXray_512_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp/log
```

## Example Training Commands

### 1. Quick Local Test (Small batch, few epochs)
```bash
python train_custom.py \
    --batch_size 2 \
    --max_epochs 5 \
    --num_workers 0 \
    --save_interval 2
```

### 2. Standard Training (Medium configuration)
```bash
python train_custom.py \
    --batch_size 8 \
    --max_epochs 100 \
    --num_workers 4 \
    --use_amp
```

### 3. High-Performance Training (Large batch, multiple GPUs)
```bash
python train_custom.py \
    --batch_size 16 \
    --max_epochs 150 \
    --num_workers 8 \
    --n_gpu 2 \
    --use_amp \
    --base_lr 0.02
```

### 4. Training with Custom Data Split
```bash
python train_custom.py \
    --train_ratio 0.9 \
    --batch_size 8 \
    --max_epochs 150
```

## Key Features

1. **Automatic Mixed Precision (AMP)**: Reduces memory usage and speeds up training on modern GPUs
2. **Validation Loop**: Monitors model performance with Dice Score after each epoch
3. **Best Model Saving**: Automatically saves the best performing model
4. **Data Augmentation**: Includes rotation and flipping for better generalization
5. **TensorBoard Integration**: Visualize training curves and predictions
6. **Flexible Configuration**: All parameters configurable via command line

## Notes

- The dataset loader automatically converts grayscale X-rays to 3-channel RGB by repeating the channel
- Masks are binarized with a threshold of 127 (values >127 become 1, else 0)
- Training uses a combination of Cross Entropy Loss and Dice Loss (0.5 * CE + 0.5 * Dice)
- Learning rate follows polynomial decay: `lr = base_lr * (1 - iter/max_iter)^0.9`
- Best model is saved based on validation Dice Score

## Troubleshooting

**Out of Memory Error:**
- Reduce `--batch_size`
- Enable `--use_amp` for mixed precision training
- Reduce `--img_size` (e.g., to 224 or 384)

**Pretrained weights not found:**
- The script will warn but continue training from scratch
- Download weights from TransUNet repository or train without pretrained weights

**Slow data loading:**
- Increase `--num_workers` (but not more than CPU cores)
- Use SSD for dataset storage
- Reduce `--img_size` if disk I/O is bottleneck
