"""
download_dataset.py
───────────────────
Downloads the badminton action classification dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/imrankhan75/badminton-action-classification
450 videos across 4 action classes.

Prerequisites
─────────────
1. Install Kaggle CLI:  pip install kaggle
2. Create API token at: https://www.kaggle.com/settings (Account → API)
3. Place kaggle.json in:
   - Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
   - Linux   : ~/.kaggle/kaggle.json

Usage
─────
    python download_dataset.py
"""

import os
import sys
import subprocess
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_DATASET = "imrankhan75/badminton-action-classification"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def check_kaggle():
    """Verify the kaggle package and credentials are available."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, check=True,
        )
        logger.info("Kaggle CLI: %s", result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "Kaggle CLI not found. Install it with:\n"
            "    pip install kaggle\n"
            "Then set up your API key from https://www.kaggle.com/settings"
        )
        sys.exit(1)

    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        logger.error(
            "Kaggle API key not found at %s\n"
            "Download it from https://www.kaggle.com/settings → API → Create New Token",
            kaggle_json,
        )
        sys.exit(1)


def download_and_extract():
    """Download the dataset zip and extract into data/."""
    zip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_download_tmp")
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Downloading dataset from Kaggle: %s", KAGGLE_DATASET)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", zip_dir],
        check=True,
    )

    # Find the downloaded zip
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
    if not zip_files:
        logger.error("No zip file found in %s after download.", zip_dir)
        sys.exit(1)

    zip_path = os.path.join(zip_dir, zip_files[0])
    logger.info("Extracting %s …", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_dir)

    # ── Normalise folder structure ─────────────────────────────────────────────
    # The Kaggle zip may have a top-level folder or not — handle both
    extracted_items = [
        os.path.join(zip_dir, item)
        for item in os.listdir(zip_dir)
        if not item.endswith(".zip")
    ]

    ACTION_MAP = {
        # Possible Kaggle folder names → our canonical names
        "backhand drive":    "backhand_drive",
        "backhand_drive":    "backhand_drive",
        "BackhandDrive":     "backhand_drive",
        "backhand net shot": "backhand_net_shot",
        "backhand_net_shot": "backhand_net_shot",
        "BackhandNetShot":   "backhand_net_shot",
        "forehand clear":    "forehand_clear",
        "forehand_clear":    "forehand_clear",
        "ForehandClear":     "forehand_clear",
        "forehand drive":    "forehand_drive",
        "forehand_drive":    "forehand_drive",
        "ForehandDrive":     "forehand_drive",
    }

    def _move_action_folders(src_root: str):
        for item in os.listdir(src_root):
            src = os.path.join(src_root, item)
            if not os.path.isdir(src):
                continue
            canonical = ACTION_MAP.get(item)
            if canonical:
                dst = os.path.join(DATA_DIR, canonical)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
                logger.info("  Moved '%s' → data/%s", item, canonical)

    # Try top-level first
    _move_action_folders(zip_dir)

    # Check if there's a nested folder
    for item in extracted_items:
        if os.path.isdir(item):
            _move_action_folders(item)

    # Cleanup temp dir
    shutil.rmtree(zip_dir, ignore_errors=True)

    # Report
    logger.info("\nDataset ready in: %s", DATA_DIR)
    for cls in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            logger.info("  %-25s : %d files", cls, count)


if __name__ == "__main__":
    check_kaggle()
    download_and_extract()
    logger.info(
        "\nNext step: python main.py --mode extract"
    )
