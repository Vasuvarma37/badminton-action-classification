"""
download_bst.py
───────────────
One-command downloader for the BST badminton skeleton datasets.

Downloads the pre-processed .npy files from the BST Google Drive links
published in: https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer

Requirements
────────────
    pip install gdown

Usage
─────
    # Download BadmintonDB (6 strokes, seq_len=72) — recommended
    python download_bst.py --source badmintondb

    # Download ShuttleSet 25-class merged (seq_len=30) — largest dataset
    python download_bst.py --source shuttleset25

    # Download both
    python download_bst.py --source all

    # List what's already downloaded
    python download_bst.py --list
"""

import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry — Google Drive file IDs from the BST README
# ─────────────────────────────────────────────────────────────────────────────
_DATASETS = {
    "badmintondb": {
        "name":    "BadmintonDB_data (seq_len=72, 6 strokes)",
        "dest":    "bst_data/BadmintonDB_data",
        "files": {
            "data.npy":  "1dlEntSV6NBAKtU7_SsXEpvUk8lK9lPTb",  # from BST README
        },
        "notes": (
            "BadmintonDB has 6 stroke categories (clear, drive, lift, "
            "net shot, serve, smash). 'serve' is discarded during loading. "
            "seq_len=72 will be resampled to FRAMES_PER_VIDEO=30."
        ),
    },
    "shuttleset25": {
        "name":    "ShuttleSet_data_merged (seq_len=30, 25 classes)",
        "dest":    "bst_data/ShuttleSet_data_merged",
        "files": {
            "data.npy":  "12Hv0abFNXeOmC4JiFtndPQz6KdCBqdHx",  # from BST README
        },
        "notes": (
            "ShuttleSet merged has 25 stroke types. ~11 of them map to your "
            "6 classes; the rest are discarded during loading. "
            "seq_len=30 matches FRAMES_PER_VIDEO — no resampling needed."
        ),
    },
    "shuttleset35": {
        "name":    "ShuttleSet_data (seq_len=30, 35 classes)",
        "dest":    "bst_data/ShuttleSet_data",
        "files": {
            "data.npy":  "1396RVvMxdUzMztiKA7leUvBJM95EF9LP",  # from BST README
        },
        "notes": "35-class variant — fewer samples per class than the merged version.",
    },
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _check_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        logger.error("gdown is not installed. Run:  pip install gdown")
        sys.exit(1)


def download_dataset(key: str) -> None:
    gdown = _check_gdown()
    info  = _DATASETS[key]
    dest  = os.path.join(_BASE_DIR, info["dest"])
    os.makedirs(dest, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading: %s", info["name"])
    logger.info("Destination: %s", dest)
    logger.info("%s", info["notes"])
    logger.info("=" * 60)

    for filename, file_id in info["files"].items():
        out_path = os.path.join(dest, filename)
        if os.path.exists(out_path):
            logger.info("  Already exists — skipping: %s", filename)
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info("  Downloading %s …", filename)
        gdown.download(url, out_path, quiet=False)
        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            logger.info("  ✓ Saved %.1f MB → %s", size_mb, out_path)
        else:
            logger.error("  ✗ Download failed for %s", filename)

    # ── Post-download: BadmintonDB needs separate label file ──────────────────
    # The BST README only shows a single data.npy for BadmintonDB.
    # If the file is a dict-format .npy it contains both data & labels;
    # the bst_adapter will handle that transparently.
    logger.info("Download complete for '%s'. Run adapter sanity check:", key)
    logger.info(
        "  python -c \"from utils.bst_adapter import load_bst_samples, BST_SOURCE; "
        "s = load_bst_samples(BST_SOURCE.%s); print(len(s), s[0][0].shape)\"",
        key.upper().replace("SHUTTLESET25", "SHUTTLESET_25")
              .replace("SHUTTLESET35", "SHUTTLESET_35"),
    )


def list_status() -> None:
    print("\n{'Dataset':<35} {'Status':<15} {'Destination'}")
    print("-" * 75)
    for key, info in _DATASETS.items():
        dest = os.path.join(_BASE_DIR, info["dest"])
        files = info["files"]
        found = sum(1 for f in files if os.path.exists(os.path.join(dest, f)))
        total = len(files)
        status = "✓ complete" if found == total else (f"{found}/{total} files" if found else "not downloaded")
        print(f"  {info['name']:<33} {status:<15} {dest}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download BST badminton skeleton datasets")
    parser.add_argument(
        "--source",
        choices=["badmintondb", "shuttleset25", "shuttleset35", "all"],
        default="badmintondb",
        help="Which dataset to download (default: badmintondb)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List download status and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_status()
        return

    if args.source == "all":
        for key in _DATASETS:
            download_dataset(key)
    else:
        download_dataset(args.source)


if __name__ == "__main__":
    main()
