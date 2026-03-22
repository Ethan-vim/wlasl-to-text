"""
Standard cross-entropy training loop for STGCNClassifier.

Uses batched training with label smoothing, mixup augmentation,
AdamW optimizer, and cosine annealing with linear warmup.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data.augment import get_ce_train_transforms, get_val_transforms, mixup_data, mixup_criterion
from src.data.dataset import WLASLKeypointDataset, get_dataloader
from src.models.classifier import build_classifier
from src.training.config import Config, load_config, save_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int = 0,
    writer: Optional[object] = None,
    global_step: int = 0,
    scheduler: Optional[object] = None,
    step_scheduler_per_batch: bool = False,
) -> tuple[float, float, int]:
    """Train for one epoch of batches.

    Parameters
    ----------
    scheduler : optional
        Learning rate scheduler.  When *step_scheduler_per_batch* is True
        the scheduler is stepped after every optimizer step (required for
        OneCycleLR).
    step_scheduler_per_batch : bool
        If True, call ``scheduler.step()`` after each batch instead of
        once per epoch.

    Returns
    -------
    tuple[float, float, int]
        (avg_loss, avg_accuracy, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    non_blocking = device.type == "cuda"
    use_mixup = cfg.mixup_alpha > 0
    use_aux = getattr(cfg, "aux_loss_weight", 0.0) > 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device, non_blocking=non_blocking)
        batch_y = batch_y.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, cfg.mixup_alpha)
            logits = model(mixed_x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        elif use_aux:
            logits, aux_logits_list = model.forward_with_aux(batch_x)
            loss = criterion(logits, batch_y)
            if aux_logits_list:
                aux_loss = sum(criterion(aux_lg, batch_y) for aux_lg in aux_logits_list) / len(aux_logits_list)
                loss = loss + cfg.aux_loss_weight * aux_loss
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Step per-batch scheduler (e.g. OneCycleLR)
        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step()

        # MPS queues Metal commands asynchronously -- flush every step to
        # prevent the command queue from filling up and stalling.
        if device.type == "mps":
            torch.mps.synchronize()

        # Batch accuracy (on unmixed labels for meaningful tracking)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == batch_y).sum().item()
            total_correct += correct
            total_samples += batch_y.size(0)

        total_loss += loss.item()
        global_step += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * total_correct / max(total_samples, 1):.1f}%",
        )

        if writer is not None and global_step % cfg.log_interval == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_acc", 100.0 * correct / batch_y.size(0), global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)

    num_batches = len(loader)
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, avg_acc, global_step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Validate on the full validation set.

    Returns
    -------
    tuple[float, float, float]
        (val_loss, top1_accuracy, top5_accuracy)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_logits: list[torch.Tensor] = []

    non_blocking = device.type == "cuda"
    for batch_x, batch_y in tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=non_blocking)
        batch_y = batch_y.to(device, non_blocking=non_blocking)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch_y.cpu().tolist())
        all_logits.append(logits.cpu())

        if device.type == "mps":
            torch.mps.synchronize()

    if not all_targets:
        return 0.0, 0.0, 0.0

    num_batches = len(loader)
    avg_loss = total_loss / max(num_batches, 1)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_logits_cat = torch.cat(all_logits, dim=0)

    # Top-1
    top1 = float(np.mean(all_preds_arr == all_targets_arr)) * 100.0

    # Top-5
    k = min(5, all_logits_cat.size(1))
    top5_preds = all_logits_cat.topk(k, dim=1).indices.numpy()
    top5_correct = np.array([
        all_targets_arr[i] in top5_preds[i] for i in range(len(all_targets_arr))
    ])
    top5 = float(np.mean(top5_correct)) * 100.0

    return avg_loss, top1, top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config, device_override: str | None = None) -> None:
    """Run the full cross-entropy training pipeline."""
    # Device
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, checkpoint_dir / "config.yaml")

    # TensorBoard
    writer = None
    if cfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))

    # W&B
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=vars(cfg) if hasattr(cfg, "__dict__") else {},
            )
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            cfg.use_wandb = False

    # Datasets
    data_dir = Path(cfg.data_dir)
    splits_dir = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}"
    processed_dir = data_dir / "processed"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}\n"
            f"Run preprocessing first:\n"
            f"  python -m src.data.preprocess --data-dir {cfg.data_dir} "
            f"--subset WLASL{cfg.wlasl_variant}"
        )

    train_transform = get_ce_train_transforms(T=cfg.T)
    val_transform = get_val_transforms(T=cfg.T)

    train_ds = WLASLKeypointDataset(
        split_csv=train_csv,
        keypoint_dir=processed_dir,
        transform=train_transform,
        T=cfg.T,
        use_motion=cfg.use_motion,
    )
    val_ds = WLASLKeypointDataset(
        split_csv=val_csv,
        keypoint_dir=processed_dir,
        transform=val_transform,
        T=cfg.T,
        use_motion=cfg.use_motion,
    )

    logger.info("Dataset: %d train / %d val samples", len(train_ds), len(val_ds))

    # DataLoaders
    use_pin_memory = device.type == "cuda"
    num_workers = cfg.num_workers if device.type != "mps" else 0

    train_loader = get_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        weighted_sampling=cfg.weighted_sampling,
    )
    # Override pin_memory for MPS
    train_loader.pin_memory = use_pin_memory

    val_loader = get_dataloader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader.pin_memory = use_pin_memory

    # Model
    model = build_classifier(cfg).to(device)

    # Loss
    if getattr(cfg, "class_weighted_loss", True):
        class_counts = np.bincount(train_ds.labels, minlength=cfg.num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)  # avoid division by zero
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * cfg.num_classes
        loss_weight = torch.from_numpy(class_weights).float().to(device)
        logger.info("Using class-weighted CE loss (weight range: %.2f - %.2f)", loss_weight.min(), loss_weight.max())
    else:
        loss_weight = None
    criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=cfg.label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Scheduler
    step_scheduler_per_batch = False
    if getattr(cfg, "scheduler", "cosine") == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.warmup_epochs / max(cfg.epochs, 1),
            anneal_strategy="cos",
        )
        step_scheduler_per_batch = True
        logger.info("Using OneCycleLR scheduler (steps_per_epoch=%d)", steps_per_epoch)
    else:
        # Linear warmup then cosine annealing (per-epoch stepping)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=max(cfg.warmup_epochs, 1),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
            eta_min=cfg.lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_epochs],
        )
        logger.info("Using cosine scheduler with linear warmup")

    # Resume
    start_epoch = 0
    best_top1 = 0.0
    global_step = 0
    if cfg.resume_checkpoint is not None:
        ckpt_path = Path(cfg.resume_checkpoint)
        if ckpt_path.exists():
            logger.info("Resuming from %s", ckpt_path)
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"]
            state_dict.pop("prototypes", None)
            model.load_state_dict(state_dict, strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_top1 = ckpt.get("best_top1", 0.0)
            global_step = ckpt.get("global_step", 0)
        else:
            logger.warning("Checkpoint not found at %s, training from scratch", ckpt_path)

    # Training loop
    epochs_without_improvement = 0
    with logging_redirect_tqdm():
        for epoch in range(start_epoch, cfg.epochs):
            t0 = time.time()

            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device, cfg,
                epoch=epoch, writer=writer, global_step=global_step,
                scheduler=scheduler, step_scheduler_per_batch=step_scheduler_per_batch,
            )

            # Step per-epoch schedulers (cosine). OneCycleLR is stepped per batch.
            if not step_scheduler_per_batch:
                scheduler.step()

            val_loss, val_top1, val_top5 = validate(
                model, val_loader, criterion, device,
            )

            elapsed = time.time() - t0

            logger.info(
                "Epoch %d/%d (%.1fs) | "
                "Train Loss: %.4f Acc: %.1f%% | "
                "Val Loss: %.4f Top1: %.1f%% Top5: %.1f%%",
                epoch + 1, cfg.epochs, elapsed,
                train_loss, train_acc,
                val_loss, val_top1, val_top5,
            )

            # TensorBoard
            if writer is not None:
                writer.add_scalar("train/loss_epoch", train_loss, epoch)
                writer.add_scalar("train/acc_epoch", train_acc, epoch)
                writer.add_scalar("val/loss_epoch", val_loss, epoch)
                writer.add_scalar("val/top1_epoch", val_top1, epoch)
                writer.add_scalar("val/top5_epoch", val_top5, epoch)

            # W&B
            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/top1": val_top1,
                    "val/top5": val_top5,
                }, step=global_step)

            # Save best
            is_best = val_top1 > best_top1
            if is_best:
                best_top1 = val_top1
                epochs_without_improvement = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_top1": best_top1,
                    "global_step": global_step,
                    "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
                }
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(ckpt, str(best_path))
                logger.info("Saved best model (Top1: %.1f%%) to %s", best_top1, best_path)
            else:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_top1": best_top1,
                    "global_step": global_step,
                }
                torch.save(ckpt, str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"))

            # Early stopping
            if (
                cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    cfg.early_stopping_patience,
                )
                break

    logger.info("Training complete. Best validation Top-1: %.1f%%", best_top1)

    if writer is not None:
        writer.close()
    if cfg.use_wandb:
        import wandb
        wandb.finish()
