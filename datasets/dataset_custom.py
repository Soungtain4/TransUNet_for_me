import os
import random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from glob import glob


def random_rot_flip(image, label):
    """Randomly rotate and flip image and label."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Randomly rotate image and label."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """Random data augmentation for training."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Convert grayscale to RGB if needed (repeat channels)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Transpose to [C, H, W] format for PyTorch
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ChestXrayDataset(Dataset):
    """
    Custom dataset for Chest X-ray lung segmentation.

    Dataset structure:
        data_root/CXR_png/CHNCXR_0001_0.png (grayscale X-ray images)
        data_root/masks/CHNCXR_0001_0_mask.png (binary masks)
    """
    def __init__(self, data_root, split='train', transform=None, train_ratio=0.7, val_ratio=0.15):
        """
        Args:
            data_root: Root directory containing CXR_png and masks subdirectories
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
            train_ratio: Ratio of training data (default 0.7)
            val_ratio:   Ratio of validation data (default 0.15); test = remainder
        """
        self.data_root = data_root
        self.transform = transform
        self.split = split

        # Get all image files
        image_dir = os.path.join(data_root, 'CXR_png')
        mask_dir = os.path.join(data_root, 'masks')

        # Find all images and their corresponding masks
        image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        all_samples = []

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            # Try both naming conventions:
            #   CHNCXR → CHNCXR_XXXX_0_mask.png  (Shenzhen)
            #   MCUCXR → MCUCXR_XXXX_0.png        (Montgomery, same name as image)
            for mask_name in [img_name.replace('.png', '_mask.png'), img_name]:
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    all_samples.append((img_path, mask_path))
                    break

        # 70 / 15 / 15 split
        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == 'train':
            self.samples = all_samples[:n_train]
        elif split == 'val':
            self.samples = all_samples[n_train:n_train + n_val]
        elif split == 'test':
            self.samples = all_samples[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        print(f"Loaded {len(self.samples)} {split} samples from {data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image (grayscale)
        image = Image.open(img_path).convert('L')
        image = np.array(image)

        # Load mask (grayscale)
        label = Image.open(mask_path).convert('L')
        label = np.array(label)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Binarize mask (assume binary segmentation: 0=background, 1=lung)
        label = (label > 127).astype(np.uint8)

        # Add channel dimension to grayscale image
        image = np.expand_dims(image, axis=2)  # [H, W, 1]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = os.path.basename(img_path).replace('.png', '')
        return sample


def get_dataloaders(data_root, batch_size=8, img_size=512, num_workers=4, train_ratio=0.8):
    """
    Convenience function to create train and validation dataloaders.

    Args:
        data_root: Root directory containing CXR_png and masks subdirectories
        batch_size: Batch size for training
        img_size: Target image size (will resize to img_size x img_size)
        num_workers: Number of worker processes for data loading
        train_ratio: Ratio of training data

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Create datasets
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
