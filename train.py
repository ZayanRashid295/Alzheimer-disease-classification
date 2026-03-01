"""
Training script: stratified split, class-weighted loss, AMP, early stopping, checkpointing, TensorBoard.
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# Add project root for imports when run as script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data import (
    get_class_names_and_splits,
    AlzheimerMRIDataset,
    get_train_transforms,
    get_eval_transforms,
)
from models import AlzheimerCNN
from utils.metrics import compute_metrics, compute_class_weights
from utils.losses import FocalLoss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_input_channels(train_pairs: list, extensions: tuple) -> int:
    """Open first image and return number of channels (1 or 3)."""
    from PIL import Image
    for path, _ in train_pairs:
        if path.suffix.lower() in extensions:
            try:
                img = Image.open(path)
                arr = np.array(img)
                ch = arr.shape[2] if arr.ndim == 3 else 1
                return min(ch, 3)
            except Exception:
                continue
    return 3


def build_dataloaders(
    data_root: Path,
    train_pairs: list,
    val_pairs: list,
    class_names: list,
    in_channels: int,
) -> tuple:
    train_ds = AlzheimerMRIDataset(
        train_pairs,
        transform=get_train_transforms(
            config.IMAGE_SIZE,
            config.AUGMENTATION,
            channels=in_channels,
        ),
        skip_corrupted=True,
    )
    val_ds = AlzheimerMRIDataset(
        val_pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=in_channels),
        skip_corrupted=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    clip_grad: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / len(y_true)
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI classifier")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT, help="Dataset root")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # Discover dataset (pre-split data/ or combined_images with stratified split)
    class_names, train_pairs, val_pairs, test_pairs = get_class_names_and_splits(
        args.data_root,
        config.IMAGE_EXTENSIONS,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_STATE,
    )
    num_classes = len(class_names)
    in_channels = config.INPUT_CHANNELS
    if in_channels is None:
        in_channels = detect_input_channels(train_pairs, config.IMAGE_EXTENSIONS)
        logger.info("Auto-detected input channels: %d", in_channels)
    logger.info(
        "Classes: %s | Train %d Val %d Test %d",
        class_names,
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    train_loader, val_loader = build_dataloaders(
        args.data_root,
        train_pairs,
        val_pairs,
        class_names,
        in_channels,
    )

    # Class weights from training set
    train_labels = [p[1] for p in train_pairs]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    if config.USE_FOCAL_LOSS or args.focal:
        criterion = FocalLoss(
            alpha=config.FOCAL_ALPHA if config.FOCAL_ALPHA is not None else class_weights,
            gamma=config.FOCAL_GAMMA,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = AlzheimerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=config.CONV_FILTERS,
        fc_sizes=config.FC_SIZES,
        dropout=config.DROPOUT,
        use_se=config.USE_SE_ATTENTION,
        he_init=config.HE_INIT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (trainable: %d)", n_params, n_trainable)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN,
    )
    use_amp = config.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.LOG_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            config.GRADIENT_CLIP_MAX_NORM,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/val_accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("metrics/val_f1", val_metrics["f1"], epoch)
        writer.add_scalar("metrics/val_precision", val_metrics["precision"], epoch)
        writer.add_scalar("metrics/val_recall", val_metrics["recall"], epoch)

        logger.info(
            "Epoch %d | Train loss %.4f | Val loss %.4f | Acc %.4f | P %.4f R %.4f F1 %.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
        )
        for i, name in enumerate(class_names):
            r = val_metrics["per_class_recall"][i]
            writer.add_scalar(f"recall/{name}", r, epoch)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "class_names": class_names,
                "in_channels": in_channels,
                "num_classes": num_classes,
            }
            torch.save(ckpt, config.CHECKPOINT_DIR / "best_model.pt")
            torch.save(ckpt, config.CHECKPOINT_DIR / f"best_model_{timestamp}.pt")
            logger.info("Saved best checkpoint (acc %.4f)", best_val_acc)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    writer.close()
    logger.info("Training finished. Best val accuracy: %.4f at epoch %d", best_val_acc, best_epoch)


if __name__ == "__main__":
    main()
