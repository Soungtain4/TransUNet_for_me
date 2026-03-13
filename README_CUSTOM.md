# TransUNet — Chest X-ray Lung Segmentation

Custom adaptation of TransUNet for 2D chest X-ray lung segmentation.

---

## Changes from Original

### 1. Legacy Compatibility Fixes
- Replaced deprecated `scipy.ndimage.interpolation.zoom` with `scipy.ndimage.zoom`
- Updated `requirements.txt` for PyTorch 2.x and NumPy >= 2.0

### 2. Custom Dataset Loader
- Added [datasets/dataset_custom.py](datasets/dataset_custom.py)
- Auto-matches X-ray images (`CXR_png/`) with masks (`masks/`)
- Converts grayscale to 3-channel pseudo-RGB `[3, 512, 512]`
- Resizes to 512×512 using bilinear interpolation (`order=1`)
- Augmentation: random rotation and horizontal/vertical flipping
- Auto train/val split (default 80/20)

### 3. Training Scripts
- [train_custom.py](train_custom.py): argparse-based CLI training script
- [run_training.ipynb](run_training.ipynb): Jupyter notebook pipeline
- AMP (Automatic Mixed Precision) support
- Best model saving based on validation Dice Score
- TensorBoard integration

---

## Available Datasets

Four datasets are available under the `data/` directory.

### 1. Chest Xray Masks and Labels (Montgomery & Shenzhen) — **Current Default**
- **Samples**: 566 matched pairs (800 images, 704 masks total)
- **Format**: PNG
- **Task**: Binary lung segmentation (background / lung)
- **Path**: `../data/Chest Xray Masks and Labels/data/Lung Segmentation/`
- **Structure**:
  ```
  Lung Segmentation/
  ├── CXR_png/    ← X-ray images  (CHNCXR_XXXX_0.png)
  └── masks/      ← Binary masks  (CHNCXR_XXXX_0_mask.png)
  ```

### 2. Chest X-ray Dataset for Tuberculosis Segmentation — **Recommended Alternative**
- **Samples**: **704 pairs** (more than the default dataset)
- **Format**: PNG
- **Task**: Binary lung segmentation (includes TB patients)
- **Path**: `../data/Chest X-ray Dataset for Tuberculosis Segmentation/Chest-X-Ray/Chest-X-Ray/`
- **Structure**:
  ```
  Chest-X-Ray/
  ├── image/    ← X-ray images
  └── mask/     ← Binary masks
  ```
- **Note**: Mask filename convention may differ — verify before use and update `dataset_custom.py` if needed.

### 3. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Samples**: 20 training / 20 validation
- **Format**: TIF (images), PNG (annotations)
- **Task**: Retinal vessel segmentation (different domain from chest X-ray)
- **Path**: `../data/DRIVE/`
- **Structure**:
  ```
  DRIVE/
  ├── images/training/
  ├── images/validation/
  ├── annotations/training/
  └── annotations/validation/
  ```

### 4. SIIM ACR Pneumothorax Segmentation Data
- **Samples**: ~24,000 DICOM files
- **Format**: DICOM + RLE-encoded CSV (`train-rle.csv`)
- **Task**: Pneumothorax segmentation
- **Path**: `../data/SIIM ACR Pneumothorax Segmentation Data/`
- **Note**: Requires DICOM-to-PNG conversion and RLE decoding before use. Not plug-and-play.

---

## Installation

```bash
cd TransUNet
pip install -r requirements.txt
```

Download pretrained weights and place at:
```
../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

---

## Usage

### Jupyter Notebook (Recommended)

Run `run_training.ipynb` top to bottom.

> **Windows users**: Set `NUM_WORKERS = 0`. Windows uses `spawn` for multiprocessing, causing significant DataLoader startup overhead when `num_workers > 0`.

### Command Line (train_custom.py)

**Quick test:**
```bash
python train_custom.py \
    --batch_size 2 \
    --max_epochs 10 \
    --num_workers 0 \
    --img_size 512
```

**Full training with AMP:**
```bash
python train_custom.py \
    --batch_size 8 \
    --max_epochs 150 \
    --num_workers 0 \
    --img_size 512 \
    --base_lr 0.01 \
    --use_amp
```

### All Arguments

**Data:**
- `--root_path`: Dataset root directory
- `--dataset`: Dataset name (default: `ChestXray`)
- `--num_classes`: Number of classes (default: `2`)
- `--train_ratio`: Train/val split ratio (default: `0.8`)

**Model:**
- `--vit_name`: ViT variant (default: `R50-ViT-B_16`)
- `--n_skip`: Number of skip connections (default: `3`)
- `--vit_patches_size`: Patch size (default: `16`)
- `--img_size`: Input resolution (default: `512`)

**Training:**
- `--max_epochs`: Max epochs (default: `150`)
- `--batch_size`: Batch size (default: `8`)
- `--base_lr`: Initial learning rate (default: `0.01`)
- `--num_workers`: DataLoader workers (**use `0` on Windows**)
- `--use_amp`: Enable Automatic Mixed Precision
- `--save_interval`: Checkpoint save frequency in epochs (default: `10`)
- `--seed`: Random seed (default: `1234`)
- `--output_dir`: Model output directory (default: `../model`)

---

## Output Structure

```
../model/TU_ChestXray_512/
└── TU_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp/
    ├── train_log.txt    ← Training log
    ├── log/             ← TensorBoard logs
    ├── best_model.pth   ← Best Dice Score checkpoint
    ├── epoch_9.pth      ← Periodic checkpoints
    └── ...
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ../model
```

---

## Design Decisions

| Item | Choice | Reason |
|------|--------|--------|
| Loss | 0.5 × CrossEntropy + 0.5 × Dice | Handles class imbalance |
| Optimizer | SGD + momentum 0.9 | Matches original TransUNet |
| LR schedule | Polynomial decay (power=0.9) | Matches original TransUNet |
| Resize | `scipy.ndimage.zoom(order=1)` | Bilinear — 4–5× faster than bicubic |
| Channel conversion | Grayscale → 3-channel repeat | Required for ImageNet pretrained weights |
| num_workers | **0 on Windows** | Avoids spawn-based process overhead |

---

## Troubleshooting

**DataLoader is very slow (Windows):**
- Set `NUM_WORKERS = 0` — most important fix on Windows
- `num_workers > 0` causes heavy spawn overhead per epoch

**Out of Memory:**
- Reduce `BATCH_SIZE` (e.g., 8 → 4)
- Enable `USE_AMP = True`
- Reduce `IMG_SIZE` (e.g., 512 → 384)

**Pretrained weights not found:**
- Training continues from random initialization with a warning
- Performance and convergence speed will be significantly worse without pretrained weights
