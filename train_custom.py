import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_custom import ChestXrayDataset, RandomGenerator


def calculate_dice_score(pred, target, num_classes=2):
    """
    Calculate Dice score for validation.

    Args:
        pred: Predicted segmentation (batch_size, H, W)
        target: Ground truth segmentation (batch_size, H, W)
        num_classes: Number of classes

    Returns:
        Average Dice score across all classes (excluding background)
    """
    dice_scores = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    for cls in range(1, num_classes):  # Skip background (class 0)
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)

        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)

        if union == 0:
            # If both pred and target are empty, dice = 1
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / (union + 1e-7)

        dice_scores.append(dice)

    return np.mean(dice_scores) if dice_scores else 0.0


def validate(model, val_loader, num_classes, use_amp=False):
    """
    Run validation and calculate average Dice score.

    Args:
        model: The segmentation model
        val_loader: Validation dataloader
        num_classes: Number of classes
        use_amp: Whether to use automatic mixed precision

    Returns:
        Average Dice score
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for sampled_batch in tqdm(val_loader, desc="Validating", ncols=70):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if use_amp:
                with autocast():
                    outputs = model(image_batch)
            else:
                outputs = model(image_batch)

            # Get predictions
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # Calculate Dice score for this batch
            dice = calculate_dice_score(preds, label_batch, num_classes)
            dice_scores.append(dice)

    avg_dice = np.mean(dice_scores)
    model.train()
    return avg_dice


def trainer_custom(args, model, snapshot_path):
    """
    Custom trainer for Chest X-ray segmentation.

    Args:
        args: Command-line arguments
        model: The TransUNet model
        snapshot_path: Path to save model checkpoints
    """
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Create datasets
    db_train = ChestXrayDataset(
        data_root=args.root_path,
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
        train_ratio=args.train_ratio
    )

    db_val = ChestXrayDataset(
        data_root=args.root_path,
        split="val",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
        train_ratio=args.train_ratio
    )

    print(f"Training set size: {len(db_train)}")
    print(f"Validation set size: {len(db_val)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    # Initialize AMP scaler if using mixed precision
    scaler = GradScaler() if args.use_amp else None

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    best_dice = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if args.use_amp:
                # Mixed precision training
                with autocast():
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update learning rate with polynomial decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            epoch_loss += loss.item()

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 20 == 0:
                logging.info(
                    f'Epoch {epoch_num}/{max_epoch}, Iteration {iter_num}: '
                    f'loss={loss.item():.4f}, loss_ce={loss_ce.item():.4f}'
                )

            # Log images periodically
            if iter_num % 50 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...].float() * 255, iter_num)
                labs = label_batch[0, ...].unsqueeze(0).float() * 255
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Validation at end of each epoch
        avg_dice = validate(model, valloader, num_classes, use_amp=args.use_amp)
        logging.info(f"Epoch {epoch_num} - Validation Dice Score: {avg_dice:.4f}")
        writer.add_scalar('validation/dice_score', avg_dice, epoch_num)

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved new best model with Dice: {best_dice:.4f}")

        # Save checkpoint every N epochs
        save_interval = args.save_interval
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved checkpoint to {save_mode_path}")

        # Save final model
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved final model to {save_mode_path}")
            iterator.close()
            break

    logging.info(f"Training finished! Best Dice Score: {best_dice:.4f}")
    writer.close()
    return "Training Finished!"


def main():
    parser = argparse.ArgumentParser(description='TransUNet Training for Chest X-ray Segmentation')

    # Data parameters
    parser.add_argument(
        '--root_path',
        type=str,
        default='../data/Chest Xray Masks and Labels/data/Lung Segmentation',
        help='Root directory containing CXR_png and masks subdirectories'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ChestXray',
        help='Dataset name'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Number of classes (2 for binary segmentation: background + lung)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data'
    )

    # Model parameters
    parser.add_argument(
        '--vit_name',
        type=str,
        default='R50-ViT-B_16',
        help='Select ViT model (R50-ViT-B_16, ViT-B_16, etc.)'
    )
    parser.add_argument(
        '--n_skip',
        type=int,
        default=3,
        help='Number of skip connections'
    )
    parser.add_argument(
        '--vit_patches_size',
        type=int,
        default=16,
        help='ViT patch size'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=512,
        help='Input image size'
    )

    # Training parameters
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=150,
        help='Maximum number of epochs to train'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.01,
        help='Base learning rate'
    )
    parser.add_argument(
        '--n_gpu',
        type=int,
        default=1,
        help='Total number of GPUs'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--deterministic',
        type=int,
        default=1,
        help='Whether to use deterministic training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed'
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Use Automatic Mixed Precision (AMP) training'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../model',
        help='Output directory for saving models'
    )

    args = parser.parse_args()

    # Set deterministic training
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create snapshot directory
    args.exp = f'TU_{args.dataset}_{args.img_size}'
    snapshot_path = os.path.join(args.output_dir, args.exp, 'TU')
    snapshot_path = snapshot_path + '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    if args.use_amp:
        snapshot_path = snapshot_path + '_amp'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    print(f"Snapshot path: {snapshot_path}")
    print(f"Using AMP: {args.use_amp}")

    # Initialize model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # Load pretrained weights
    try:
        net.load_from(weights=np.load(config_vit.pretrained_path))
        print(f"Loaded pretrained weights from {config_vit.pretrained_path}")
    except FileNotFoundError:
        print(f"Warning: Pretrained weights not found at {config_vit.pretrained_path}")
        print("Training from scratch...")

    # Start training
    trainer_custom(args, net, snapshot_path)


if __name__ == "__main__":
    main()
