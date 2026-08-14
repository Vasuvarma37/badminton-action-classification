"""
main.py — Unified entry point for the badminton action classification pipeline.

Usage
-----
    # Step 1: Extract skeleton keypoints from raw videos
    python main.py --mode extract

    # Step 2: Download BST skeleton data
    python download_bst.py --source all   (or just badmintondb)

    # Step 3: Train (MediaPipe only)
    python main.py --mode train

    # Step 3: Train (MediaPipe + BST combined)
    python main.py --mode train --use-bst

    # Step 4: Evaluate on the test set
    python main.py --mode evaluate

    # Run the complete pipeline end-to-end (with BST)
    python main.py --mode all --use-bst

    # Sanity-check: verify the model forward pass
    python main.py --mode sanity

Flags
-----
    --overwrite     Re-extract keypoints even if .npy files already exist
    --epochs N      Override number of training epochs
    --lr F          Override learning rate
    --checkpoint    Name of checkpoint file to save/load (default: best_bilstm.pth)
    --use-bst       Incorporate BST skeleton dataset into training
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Badminton Action Classification — BiLSTM + MediaPipe + BST"
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "train", "evaluate", "all", "sanity", "bst-check"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy keypoint files during extraction",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override LEARNING_RATE from config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_bilstm.pth",
        help="Checkpoint filename (default: best_bilstm.pth)",
    )
    parser.add_argument(
        "--use-bst",
        action="store_true",
        default=None,
        help="Merge BST skeleton data into training (overrides config USE_BST)",
    )
    parser.add_argument(
        "--no-bst",
        action="store_true",
        help="Disable BST data even if USE_BST=True in config",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def run_extract(overwrite: bool = False):
    logger.info("=" * 55)
    logger.info("  STAGE 1: Keypoint Extraction (MediaPipe)")
    logger.info("=" * 55)
    from keypoint_extraction.mediapipe_extractor import run_extraction
    run_extraction(overwrite=overwrite)


def run_train(epochs=None, lr=None, checkpoint="best_bilstm.pth", use_bst=None):
    logger.info("=" * 55)
    logger.info("  STAGE 2: Training BiLSTM + Attention")
    logger.info("=" * 55)

    import config
    kwargs = {}
    if epochs is not None:
        kwargs["num_epochs"] = epochs
        logger.info("Override epochs → %d", epochs)
    if lr is not None:
        kwargs["learning_rate"] = lr
        logger.info("Override lr → %f", lr)
    if use_bst is not None:
        kwargs["use_bst"] = use_bst
        logger.info("BST data: %s", "enabled" if use_bst else "disabled")

    from model.train import train
    from utils.visualize import plot_training_curves

    history = train(checkpoint_name=checkpoint, **kwargs)
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
    )


def run_evaluate(checkpoint="best_bilstm.pth"):
    logger.info("=" * 55)
    logger.info("  STAGE 3: Evaluation on Test Set")
    logger.info("=" * 55)
    from model.evaluate import evaluate
    metrics = evaluate(checkpoint_name=checkpoint)
    return metrics


def run_sanity():
    """Quick smoke-test: model forward pass without any data."""
    logger.info("Running sanity check …")
    import torch
    from model.bilstm_attention import BadmintonBiLSTM
    from config import INPUT_SIZE, FRAMES_PER_VIDEO, NUM_CLASSES

    model = BadmintonBiLSTM()
    model.eval()
    dummy = torch.randn(4, FRAMES_PER_VIDEO, INPUT_SIZE)
    with torch.no_grad():
        logits, attn = model(dummy, return_attention=True)
    logger.info("Input  : %s", list(dummy.shape))
    logger.info("Output : %s", list(logits.shape))
    logger.info("Attn   : %s", list(attn.shape))
    logger.info("Params : %d", model.count_parameters())
    assert logits.shape == (4, NUM_CLASSES), f"Expected (4,{NUM_CLASSES}), got {logits.shape}"
    logger.info("[OK] Sanity check passed!")


def run_bst_check():
    """Verify BST adapter can load downloaded data."""
    logger.info("=" * 55)
    logger.info("  BST Data Adapter Check")
    logger.info("=" * 55)
    from utils.bst_adapter import load_bst_samples, BST_SOURCE
    from config import BST_DATA_DIR
    import os

    for src in BST_SOURCE:
        src_dir = os.path.join(BST_DATA_DIR, src.value)
        if not os.path.isdir(src_dir):
            logger.info("  [SKIP] %s — not downloaded", src.value)
            continue
        try:
            samples = load_bst_samples(source=src)
            seq, lbl = samples[0]
            logger.info(
                "  [OK]   %s — %d samples | seq shape %s",
                src.value, len(samples), seq.shape,
            )
        except Exception as e:
            logger.error("  [FAIL] %s — %s", src.value, e)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    mode = args.mode

    # Resolve BST flag: CLI args override config
    use_bst = None
    if args.use_bst:
        use_bst = True
    elif args.no_bst:
        use_bst = False
    # else: None → train() uses config.USE_BST

    if mode == "sanity":
        run_sanity()

    elif mode == "bst-check":
        run_bst_check()

    elif mode == "extract":
        run_extract(overwrite=args.overwrite)

    elif mode == "train":
        run_train(epochs=args.epochs, lr=args.lr, checkpoint=args.checkpoint, use_bst=use_bst)

    elif mode == "evaluate":
        run_evaluate(checkpoint=args.checkpoint)

    elif mode == "all":
        run_extract(overwrite=args.overwrite)
        run_train(epochs=args.epochs, lr=args.lr, checkpoint=args.checkpoint, use_bst=use_bst)
        metrics = run_evaluate(checkpoint=args.checkpoint)
        print("\n── Final Results ──────────────────────────────────────────")
        print(f"  Overall Test Accuracy : {metrics['overall_accuracy']:.2%}")
        print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
