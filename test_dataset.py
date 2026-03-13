"""
Quick test script to verify the custom Chest X-ray dataset loads correctly.
Run this before starting training to catch any issues early.
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.dataset_custom import ChestXrayDataset, RandomGenerator, get_dataloaders


def visualize_samples(dataset, num_samples=4, save_path='dataset_samples.png'):
    """Visualize random samples from the dataset."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx, ax_row in enumerate(axes):
        sample = dataset[indices[idx]]
        image = sample['image']
        label = sample['label']
        case_name = sample['case_name']

        # Convert tensor to numpy for visualization
        if torch.is_tensor(image):
            image_np = image.numpy()
            if image_np.shape[0] == 3:  # [C, H, W] format
                image_np = image_np[0]  # Take first channel for grayscale
        else:
            image_np = image

        if torch.is_tensor(label):
            label_np = label.numpy()
        else:
            label_np = label

        # Plot original image (first channel)
        ax_row[0].imshow(image_np, cmap='gray')
        ax_row[0].set_title(f'Image: {case_name}')
        ax_row[0].axis('off')

        # Plot mask
        ax_row[1].imshow(label_np, cmap='gray')
        ax_row[1].set_title('Mask')
        ax_row[1].axis('off')

        # Plot overlay
        ax_row[2].imshow(image_np, cmap='gray', alpha=0.7)
        ax_row[2].imshow(label_np, cmap='Reds', alpha=0.3)
        ax_row[2].set_title('Overlay')
        ax_row[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def test_dataset(data_root, img_size=512, train_ratio=0.8):
    """Test dataset loading and print statistics."""
    print("=" * 60)
    print("Testing Custom Chest X-ray Dataset")
    print("=" * 60)

    # Check if directories exist
    cxr_dir = os.path.join(data_root, 'CXR_png')
    mask_dir = os.path.join(data_root, 'masks')

    print(f"\nData root: {data_root}")
    print(f"CXR directory: {cxr_dir}")
    print(f"Mask directory: {mask_dir}")

    if not os.path.exists(cxr_dir):
        print(f"ERROR: CXR directory not found at {cxr_dir}")
        return False

    if not os.path.exists(mask_dir):
        print(f"ERROR: Mask directory not found at {mask_dir}")
        return False

    print("\n✓ Directories exist")

    # Create datasets
    print(f"\nCreating datasets with train_ratio={train_ratio}...")

    try:
        train_dataset = ChestXrayDataset(
            data_root=data_root,
            split='train',
            transform=RandomGenerator(output_size=[img_size, img_size]),
            train_ratio=train_ratio
        )

        val_dataset = ChestXrayDataset(
            data_root=data_root,
            split='val',
            transform=RandomGenerator(output_size=[img_size, img_size]),
            train_ratio=train_ratio
        )

        print("\n✓ Datasets created successfully")

    except Exception as e:
        print(f"\nERROR creating datasets: {e}")
        return False

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"  Image size: {img_size}x{img_size}")

    # Test loading a sample
    print("\nTesting sample loading...")
    try:
        sample = train_dataset[0]
        image = sample['image']
        label = sample['label']
        case_name = sample['case_name']

        print(f"\n✓ Successfully loaded sample: {case_name}")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Label shape: {label.shape}")
        print(f"  Label dtype: {label.dtype}")
        print(f"  Label unique values: {torch.unique(label).tolist()}")

        # Verify expected shapes
        expected_image_shape = (3, img_size, img_size)
        expected_label_shape = (img_size, img_size)

        if image.shape != expected_image_shape:
            print(f"\nWARNING: Image shape {image.shape} doesn't match expected {expected_image_shape}")
        else:
            print(f"  ✓ Image shape is correct: {expected_image_shape}")

        if label.shape != expected_label_shape:
            print(f"\nWARNING: Label shape {label.shape} doesn't match expected {expected_label_shape}")
        else:
            print(f"  ✓ Label shape is correct: {expected_label_shape}")

    except Exception as e:
        print(f"\nERROR loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataloader
    print("\nTesting DataLoader...")
    try:
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )

        batch = next(iter(train_loader))
        print(f"\n✓ DataLoader works")
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch label shape: {batch['label'].shape}")

    except Exception as e:
        print(f"\nERROR with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Visualize samples
    print("\nGenerating visualization...")
    try:
        visualize_samples(train_dataset, num_samples=4, save_path='train_samples.png')
        visualize_samples(val_dataset, num_samples=2, save_path='val_samples.png')
        print("✓ Visualization complete")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Dataset is ready for training!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Custom Chest X-ray Dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='../data/Chest Xray Masks and Labels/data/Lung Segmentation',
        help='Root directory containing CXR_png and masks subdirectories'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=512,
        help='Target image size'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data'
    )

    args = parser.parse_args()

    success = test_dataset(args.root_path, args.img_size, args.train_ratio)

    if not success:
        print("\n❌ Tests failed. Please fix the issues before training.")
        exit(1)
    else:
        print("\nYou can now start training with:")
        print("  python train_custom.py --batch_size 8 --max_epochs 150 --use_amp")


if __name__ == "__main__":
    main()
