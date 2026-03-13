# Quick Start Guide - Chest X-ray Segmentation

This guide will get you started with training TransUNet on your Chest X-ray dataset in just a few steps.

## Step 1: Install Dependencies

```bash
cd TransUNet
pip install -r requirements.txt
```

## Step 2: Test Dataset Loading

Before training, verify your dataset loads correctly:

```bash
python test_dataset.py
```

This will:
- Check if your data directories exist
- Load sample images and masks
- Verify shapes and data types
- Create visualizations (`train_samples.png` and `val_samples.png`)

**Expected output:**
```
✓ ALL TESTS PASSED - Dataset is ready for training!
```

## Step 3: Start Training

### Option A: Quick Local Test (Recommended First)
Test with small batch size and few epochs to verify everything works:

```bash
python train_custom.py \
    --batch_size 2 \
    --max_epochs 5 \
    --num_workers 0 \
    --save_interval 2
```

### Option B: Full Training Session
Once verified, run full training with optimizations:

```bash
python train_custom.py \
    --batch_size 8 \
    --max_epochs 150 \
    --num_workers 4 \
    --use_amp \
    --base_lr 0.01 \
    --save_interval 10
```

### Option C: High-Performance Training (Multiple GPUs)
For faster training on powerful servers:

```bash
python train_custom.py \
    --batch_size 16 \
    --max_epochs 150 \
    --num_workers 8 \
    --n_gpu 2 \
    --use_amp \
    --base_lr 0.02
```

## Step 4: Monitor Training

### View Logs
```bash
# Watch training progress in real-time
tail -f ../model/TU_ChestXray_512_*/log.txt
```

### TensorBoard Visualization
```bash
# Start TensorBoard
tensorboard --logdir ../model/TU_ChestXray_512_*/log

# Open browser to http://localhost:6006
```

You'll see:
- Training/validation loss curves
- Learning rate schedule
- Dice score over epochs
- Sample predictions vs ground truth

## What to Expect

### Training Output Location
```
../model/TU_ChestXray_512_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp/
├── log.txt              # Text logs
├── log/                 # TensorBoard logs
├── best_model.pth       # Best model (highest validation Dice)
├── epoch_9.pth          # Checkpoint every N epochs
├── epoch_19.pth
└── epoch_149.pth        # Final model
```

### During Training
- **Logs every 20 iterations**: Loss values
- **Validation after each epoch**: Dice score
- **Auto-saves best model**: Based on validation Dice
- **Checkpoints every N epochs**: Configurable with `--save_interval`

### Typical Training Time
- **Local CPU**: Very slow, not recommended
- **Single GPU (e.g., RTX 3090)**: ~2-4 hours for 150 epochs
- **Multi-GPU setup**: ~1-2 hours with parallel training

## Common Issues & Solutions

### Issue 1: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
python train_custom.py \
    --batch_size 4 \        # Reduce batch size
    --use_amp \             # Enable mixed precision
    --img_size 384          # Use smaller image size
```

### Issue 2: Dataset Not Found
```
ERROR: CXR directory not found
```

**Solution:**
Verify your data path:
```bash
python train_custom.py \
    --root_path "/path/to/your/data/Lung Segmentation"
```

### Issue 3: Slow Training
**Solution:**
- Increase `--num_workers` (e.g., 4-8)
- Enable `--use_amp` for faster computation
- Use SSD instead of HDD for dataset storage
- Reduce `--img_size` if acceptable

### Issue 4: Pretrained Weights Not Found
```
Warning: Pretrained weights not found
```

**Solution:**
- Download from TransUNet repository, or
- Continue training from scratch (will take longer to converge)

## Next Steps

After training completes:

1. **Check best model performance:**
   Look for `Best Dice Score: X.XXXX` in logs

2. **Load and test model:**
   ```python
   import torch
   from networks.vit_seg_modeling import VisionTransformer as ViT_seg
   from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

   config = CONFIGS_ViT_seg['R50-ViT-B_16']
   config.n_classes = 2
   model = ViT_seg(config, img_size=512, num_classes=2)
   model.load_state_dict(torch.load('path/to/best_model.pth'))
   model.eval()
   ```

3. **Visualize predictions:**
   Check TensorBoard for visual results during training

4. **Fine-tune hyperparameters:**
   Adjust `--base_lr`, `--batch_size`, `--max_epochs` based on results

## Recommended Workflow

1. **Day 1: Setup & Verification**
   ```bash
   pip install -r requirements.txt
   python test_dataset.py
   python train_custom.py --batch_size 2 --max_epochs 2  # Quick test
   ```

2. **Day 2: Short Training Run**
   ```bash
   python train_custom.py --batch_size 8 --max_epochs 30 --use_amp
   ```
   Monitor for issues, check if loss decreases

3. **Day 3+: Full Training**
   ```bash
   python train_custom.py --batch_size 8 --max_epochs 150 --use_amp
   ```
   Let it run overnight or over weekend

## Key Command-Line Arguments

| Argument | Default | Description | When to Change |
|----------|---------|-------------|----------------|
| `--batch_size` | 8 | Batch size per GPU | Reduce if OOM, increase if underutilizing GPU |
| `--max_epochs` | 150 | Number of epochs | Reduce for quick tests |
| `--use_amp` | False | Enable mixed precision | Always use on modern GPUs |
| `--base_lr` | 0.01 | Learning rate | Try 0.001-0.02 range |
| `--num_workers` | 4 | Data loading workers | Match to CPU cores |
| `--img_size` | 512 | Input image size | Reduce if OOM |
| `--train_ratio` | 0.8 | Train/val split | Adjust based on dataset size |
| `--save_interval` | 10 | Checkpoint frequency | Reduce for more frequent saves |

## Tips for Best Results

1. **Always test first**: Run `test_dataset.py` before training
2. **Use AMP**: `--use_amp` for faster training and lower memory
3. **Monitor early**: Check TensorBoard after first epoch
4. **Start small**: Test with 2-5 epochs before full training
5. **Save disk space**: Adjust `--save_interval` to avoid too many checkpoints

## Help & Documentation

- Full documentation: [README_CUSTOM.md](README_CUSTOM.md)
- Original TransUNet: [README.md](README.md)
- Test dataset: `python test_dataset.py --help`
- Training options: `python train_custom.py --help`

Good luck with your training! 🚀
