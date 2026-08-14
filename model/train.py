"""
model/train.py
──────────────
Training script for the BadmintonBiLSTM model.

Features
--------
• Adam optimiser with weight decay (L2 regularisation)
• ReduceLROnPlateau scheduler (halves LR on val-loss plateau)
• Early stopping (stops if val accuracy doesn't improve for N epochs)
• Best-model checkpointing (saves the epoch with highest val accuracy)
• Per-class weighted CrossEntropyLoss (boost backhand_drive / weaker classes)
• WeightedRandomSampler in training loader (balanced mini-batches)
• Optional BST dataset augmentation alongside MediaPipe data

Usage
-----
    python model/train.py                   # defaults from config.py
    python main.py --mode train             # via unified entry point
    python main.py --mode train --use-bst   # include BST data
"""

import os
import sys
import json
import time
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_SAVE_DIR, RESULTS_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PAT,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FC_HIDDEN,
    NUM_CLASSES, ACTION_CLASSES, CLASS_WEIGHTS,
    USE_BST, BST_DATA_DIR,
)
from model.bilstm_attention import BadmintonBiLSTM
from utils.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("CUDA not available — training on CPU.")
    return torch.device("cpu")


def _build_class_weight_tensor(device: torch.device) -> torch.Tensor:
    """Return a FloatTensor of per-class loss weights on the target device."""
    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    return w


def _load_bst_train_samples() -> list:
    """
    Load BST samples (BadmintonDB + ShuttleSet25) and return them as
    in-memory (array, label) tuples for injection into the training split.
    Returns empty list if BST data is unavailable or USE_BST is False.
    """
    if not USE_BST:
        return []

    from utils.bst_adapter import load_bst_samples, BST_SOURCE
    samples = []

    for src, name in [
        (BST_SOURCE.BADMINTONDB,   "BadmintonDB"),
        (BST_SOURCE.SHUTTLESET_25, "ShuttleSet-25"),
    ]:
        src_dir = os.path.join(BST_DATA_DIR, src.value)
        if not os.path.isdir(src_dir):
            logger.info("BST source not found, skipping: %s", name)
            continue
        try:
            s = load_bst_samples(source=src)
            samples.extend(s)
            logger.info("Loaded %d samples from %s", len(s), name)
        except Exception as e:
            logger.warning("Could not load BST %s: %s", name, e)

    return samples


def _evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one pass over loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * seqs.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += seqs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def _bar(frac: float, width: int = 23) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def class_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Return per-class accuracy dict after evaluating on loader."""
    model.eval()
    correct_per = [0] * NUM_CLASSES
    total_per   = [0] * NUM_CLASSES

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            preds = model(seqs).argmax(dim=1)
            for c in range(NUM_CLASSES):
                mask = labels == c
                correct_per[c] += (preds[mask] == c).sum().item()
                total_per[c]   += mask.sum().item()

    result = {}
    for c, name in enumerate(ACTION_CLASSES):
        acc = correct_per[c] / max(total_per[c], 1)
        result[name] = acc
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay:  float = WEIGHT_DECAY,
    checkpoint_name: str = "best_bilstm.pth",
    use_bst: bool        = USE_BST,
) -> dict:
    """
    Train the BadmintonBiLSTM model.

    Parameters
    ----------
    use_bst : bool
        If True, load available BST skeleton data and merge it into training.

    Returns
    -------
    history : dict  — train_loss, val_loss, train_acc, val_acc per epoch
    """
    device = _get_device()

    # ── Load BST samples (if requested) ───────────────────────────────────────
    bst_samples = _load_bst_train_samples() if use_bst else []
    if bst_samples:
        logger.info("Injecting %d BST samples into training split.", len(bst_samples))

    # ── Data loaders ──────────────────────────────────────────────────────────
    logger.info("Building dataloaders …")
    train_loader, val_loader, _ = get_dataloaders(
        augment_train=True,
        extra_train_samples=bst_samples,
        use_weighted_sampler=True,
    )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BadmintonBiLSTM(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
        fc_hidden   = FC_HIDDEN,
    ).to(device)

    n_params = model.count_parameters()
    logger.info("Training — %d classes | %d params", NUM_CLASSES, n_params)
    logger.info("Max: %d epochs | Early stop: %d", num_epochs, EARLY_STOP_PAT)
    logger.info("Train samples: %d | Val samples: %d", n_train, n_val)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    class_w   = _build_class_weight_tensor(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, verbose=False,
    )

    # ── State ─────────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = os.path.join(MODEL_SAVE_DIR, checkpoint_name)
    start_time       = time.time()

    print(f"\nTraining — {NUM_CLASSES} classes | {n_params:,} params")
    print(f"Max: {num_epochs} epochs | Early stop: {EARLY_STOP_PAT}\n")

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:>3}/{num_epochs}", leave=False, ncols=90)
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * seqs.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += seqs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(total, 1)
        train_acc  = correct / max(total, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        gap    = train_acc - val_acc

        # Print every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            sign = "+" if gap >= 0 else ""
            print(
                f"Ep {epoch:>3}/{num_epochs} | "
                f"Train {train_loss:.4f} ({train_acc*100:.1f}%) | "
                f"Val {val_loss:.4f} ({val_acc*100:.1f}%) | "
                f"Gap {sign}{gap*100:.1f}% | LR {lr_now:.2e}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":            val_acc,
                    "val_loss":           val_loss,
                    "history":            history,
                },
                checkpoint_path,
            )
            print(f"  ⭐ Best → {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= EARLY_STOP_PAT:
            print(f"\nEarly stop at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n✅ Done in {elapsed/60:.1f} min | Best Val Acc: {best_val_acc*100:.2f}%")

    # ── Per-class accuracy on val set ─────────────────────────────────────────
    # Reload best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    per_class = class_accuracy(model, val_loader, device)
    print("\nPer-class accuracy:")
    for name, acc in per_class.items():
        print(f"  {name:<26} {_bar(acc)} {acc*100:.1f}%")

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = train()
    from utils.visualize import plot_training_curves
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
    )
