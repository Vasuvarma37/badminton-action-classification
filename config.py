"""
config.py  Central configuration for all hyperparameters and paths.
All scripts import from here; change values here to affect the entire pipeline.
"""

import os

# 
# Paths
# 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR       = os.path.join(BASE_DIR, "data")               # raw video folders
KEYPOINTS_DIR  = os.path.join(BASE_DIR, "extracted_keypoints") # saved .npy files
MODEL_SAVE_DIR = os.environ.get(
    "MODEL_SAVE_DIR",
    os.path.join(BASE_DIR, "model", "saved_models"),
)
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# BST external dataset  download via:  python download_bst.py
BST_DATA_DIR = os.path.join(BASE_DIR, "bst_data")
USE_BST      = False     # False  train on your MediaPipe data only

# 
# Dataset  6-class problem
# 
ACTION_CLASSES = [
    "backhand_drive",       # 0
    "backhand_net_shot",    # 1
    "forehand_clear",       # 2
    "forehand_drive",       # 3
]
NUM_CLASSES = len(ACTION_CLASSES)   # 6

#  Joint representation (common 13-joint skeleton) 
# Both MediaPipe-33 and BST/COCO-17 are projected onto the same 13 joints:
#   [L-shoulder, R-shoulder, L-elbow, R-elbow, L-wrist, R-wrist,
#    L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose/head]
# See utils/joint_mapping.py for the exact index tables.
NUM_COMMON_JOINTS  = 13
FEATURES_PER_FRAME = NUM_COMMON_JOINTS * 2   # x + y    26 features per frame

# Legacy constant (kept for reference; actual extraction still uses 33 landmarks)
NUM_LANDMARKS = 33   # MediaPipe raw count

#  Frame sampling 
# BST ShuttleSet uses seq_len=30; we match it to allow direct mixing.
# (Previous value was 10  too short to capture full stroke arc)
FRAMES_PER_VIDEO = 30

#  Train / val / test split 
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# 
# Model Hyperparameters
# 
INPUT_SIZE   = FEATURES_PER_FRAME   # 26
HIDDEN_SIZE  = 256                  # per-direction LSTM units (up from 128)
NUM_LAYERS   = 3                    # Stacked BiGRU layers
DROPOUT      = 0.4                  # slightly more regularisation than before
FC_HIDDEN    = 128                  # FC intermediate dim (up from 64)

# 
# Training
# 
BATCH_SIZE     = 64                 # larger batch for the combined dataset
NUM_EPOCHS     = 500
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4              # L2 regularisation
LR_PATIENCE    = 20                # ReduceLROnPlateau patience (epochs)
LR_FACTOR      = 0.5              # LR multiplier on plateau
EARLY_STOP_PAT = 80               # Early stopping patience (epochs)

# Per-class loss weights  order matches ACTION_CLASSES above.
# Higher weight = penalise misclassifications more for that class.
CLASS_WEIGHTS = [
    1.4,   # backhand_drive    (111 samples, fewest)
    0.9,   # backhand_net_shot (168 samples, most)
    1.2,   # forehand_clear   (119 samples)
    1.0,   # forehand_drive   (159 samples)
]

# 
# Misc
# 
NUM_WORKERS = 0    # DataLoader workers (0 = main process, safe on Windows)
PIN_MEMORY  = False
