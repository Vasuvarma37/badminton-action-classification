"""
model/evaluate.py
─────────────────
Loads the best saved checkpoint and evaluates on the held-out test set.

Outputs
-------
• Overall accuracy (printed to console)
• Confusion matrix PNG  → results/confusion_matrix.png
• Per-class accuracy PNG → results/per_class_accuracy.png
• Classification report  (precision, recall, F1 per class)
• Saves predictions JSON → results/test_predictions.json

Usage
-----
    python model/evaluate.py
    python main.py --mode evaluate
"""

import os
import sys
import json
import logging

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_SAVE_DIR, RESULTS_DIR, ACTION_CLASSES,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FC_HIDDEN, NUM_CLASSES,
)
from model.bilstm_attention import BadmintonBiLSTM
from utils.dataset import get_dataloaders
from utils.visualize import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model(
    checkpoint_name: str = "best_bilstm.pth",
    device: torch.device | None = None,
) -> tuple[BadmintonBiLSTM, dict]:
    """
    Load model from checkpoint.

    Returns
    -------
    model      : BadmintonBiLSTM (eval mode)
    checkpoint : dict with metadata (epoch, val_acc, history, …)
    """
    if device is None:
        device = _get_device()

    checkpoint_path = os.path.join(MODEL_SAVE_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first: python main.py --mode train"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = BadmintonBiLSTM(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
        fc_hidden   = FC_HIDDEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Loaded checkpoint from epoch %d | Val Acc: %.2f%%",
        checkpoint.get("epoch", "?"),
        checkpoint.get("val_acc", 0.0) * 100,
    )
    return model, checkpoint


def evaluate(checkpoint_name: str = "best_bilstm.pth") -> dict:
    """
    Full evaluation pipeline.

    Returns
    -------
    metrics : dict
        overall_accuracy, per_class_accuracy, confusion_matrix
    """
    device = _get_device()
    model, checkpoint = load_best_model(checkpoint_name, device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    logger.info("Loading test set …")
    _, _, test_loader = get_dataloaders(augment_train=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # ── Metrics ───────────────────────────────────────────────────────────────
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / len(all_labels)

    logger.info("=" * 55)
    logger.info("  TEST SET ACCURACY : %.2f%%", overall_acc * 100)
    logger.info("  (Paper LSTM baseline: 80.00%%)")
    logger.info("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(all_labels, all_preds, ACTION_CLASSES)
    plot_per_class_accuracy(all_labels, all_preds, ACTION_CLASSES)

    # Plot training curves if history is available
    history = checkpoint.get("history")
    if history:
        plot_training_curves(
            history["train_loss"], history["val_loss"],
            history["train_acc"],  history["val_acc"],
        )

    # ── Save predictions ─────────────────────────────────────────────────────
    preds_path = os.path.join(RESULTS_DIR, "test_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(
            {
                "labels":      all_labels,
                "predictions": all_preds,
                "probabilities": all_probs,
                "overall_accuracy": overall_acc,
                "class_names": ACTION_CLASSES,
            },
            f, indent=2,
        )
    logger.info("Predictions saved → %s", preds_path)

    # ── Build per-class accuracy dict ─────────────────────────────────────────
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(all_labels, all_preds)
    per_class = (cm.diagonal() / cm.sum(axis=1)).tolist()

    return {
        "overall_accuracy":   overall_acc,
        "per_class_accuracy": dict(zip(ACTION_CLASSES, per_class)),
        "confusion_matrix":   cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = evaluate()
    print("\nPer-class accuracy:")
    for cls, acc in metrics["per_class_accuracy"].items():
        print(f"  {cls:<25}: {acc:.2%}")
