# Badminton Action Classification — BiLSTM + MediaPipe

> **Paper**: *Badminton Action Classification Based on Human Skeleton Data Extracted by AlphaPose* (ICSMD 2023)
> **Improvements**: AlphaPose → MediaPipe | LSTM → BiLSTM + Attention | 80% → 85%+ target

---

## 🏸 What This Project Does

Classifies badminton strokes from video clips using **skeleton keypoint sequences**:

| Class | Description |
|---|---|
| `backhand_drive` | Flat fast shot played backhand |
| `backhand_net_shot` | Delicate backhand net touch |
| `forehand_clear` | High deep overhead clear |
| `forehand_drive` | Flat fast forehand shot |

**Pipeline**: Video → MediaPipe skeleton extraction → BiLSTM + Attention → Action label

---

## ✨ Improvements Over the Paper

| Paper | This Project |
|---|---|
| AlphaPose (complex, CUDA-heavy) | **MediaPipe Pose** (pip install, CPU-friendly) |
| 17 keypoints → 51 features | **33 keypoints → 66 features** (richer spatial info) |
| Uni-directional LSTM | **BiLSTM** (forward + backward temporal context) |
| No attention | **Bahdanau Attention** (focuses on key frames) |
| 80% accuracy | **Targeting 85%+** |
| No augmentation | **Data augmentation** (noise, flip, scale) |
| No regularisation | **Label smoothing + gradient clipping + weight decay** |

---

## 📁 Project Structure

```
badminton-action-classification/
│
├── data/                          ← Place your video folders here
│   ├── backhand_drive/
│   ├── backhand_net_shot/
│   ├── forehand_clear/
│   └── forehand_drive/
│
├── extracted_keypoints/           ← Auto-generated .npy files
│
├── model/                         ← ALL model code (separate folder)
│   ├── __init__.py
│   ├── bilstm_attention.py        ← BiLSTM + Attention architecture
│   ├── train.py                   ← Training loop
│   ├── evaluate.py                ← Evaluation + metrics
│   └── saved_models/              ← best_bilstm.pth checkpoint
│
├── keypoint_extraction/
│   └── mediapipe_extractor.py     ← MediaPipe skeleton extraction
│
├── utils/
│   ├── dataset.py                 ← PyTorch Dataset + DataLoaders
│   └── visualize.py               ← Training curves + confusion matrix
│
├── notebooks/
│   └── colab_demo.ipynb           ← Google Colab demo (free GPU)
│
├── results/                       ← Auto-generated plots + predictions
├── config.py                      ← All hyperparameters
├── main.py                        ← Unified CLI entry point
├── download_dataset.py            ← Kaggle dataset downloader
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Dataset

**Option A**: Download via Kaggle CLI (recommended)
```bash
# First set up your Kaggle API key from https://www.kaggle.com/settings
pip install kaggle
python download_dataset.py
```

**Option B**: Manual download
1. Go to https://www.kaggle.com/datasets/imrankhan75/badminton-action-classification
2. Download and extract into `data/` with the folder structure above

### 3. Run the Full Pipeline

```bash
# End-to-end (extract → train → evaluate)
python main.py --mode all

# Or step by step:
python main.py --mode extract    # Step 1: Extract keypoints
python main.py --mode train      # Step 2: Train BiLSTM
python main.py --mode evaluate   # Step 3: Evaluate on test set
```

### 4. Sanity Check (no data needed)

```bash
python main.py --mode sanity
```

---

## 🧠 Model Architecture

```
Input (B, 10, 66)
    │
LayerNorm
    │
BiLSTM Layer 1  (hidden=128, bidirectional → 256)
    │ Dropout(0.3)
BiLSTM Layer 2  (hidden=128, bidirectional → 256)
    │ Dropout(0.3)
BiLSTM Layer 3  (hidden=128, bidirectional → 256)
    │
Bahdanau Attention  (weighted sum over 10 time steps)
    │
FC(256→64) + GELU + Dropout(0.3)
    │
FC(64→4)
    │
Softmax → Action label
```

**Parameters**: ~650K (lightweight enough to train on free Colab GPU in ~15 minutes)

---

## ⚙️ Hyperparameters

All hyperparameters are in [`config.py`](config.py):

| Parameter | Value | Note |
|---|---|---|
| `FRAMES_PER_VIDEO` | 10 | Same as paper |
| `INPUT_SIZE` | 66 | 33 landmarks × (x,y) |
| `HIDDEN_SIZE` | 128 | Per BiLSTM direction |
| `NUM_LAYERS` | 3 | Stacked BiLSTM |
| `DROPOUT` | 0.3 | |
| `BATCH_SIZE` | 32 | |
| `NUM_EPOCHS` | 150 | With early stopping |
| `LEARNING_RATE` | 0.001 | |
| `EARLY_STOP_PAT` | 25 | Epochs without improvement |

---

## 📊 Expected Results

| Model | Test Accuracy |
|---|---|
| CNN (paper) | 60% |
| LSTM (paper) | 80% |
| **BiLSTM + Attention (ours)** | **85%+** |

Results saved to `results/`:
- `training_curves.png` — Loss and accuracy over epochs
- `confusion_matrix.png` — Raw + normalised confusion matrix
- `per_class_accuracy.png` — Per-class accuracy bar chart
- `test_predictions.json` — All predictions + probabilities

---

## ☁️ Google Colab

Open `notebooks/colab_demo.ipynb` in Google Colab for free T4 GPU training.
The notebook includes all steps: dataset download → extraction → training → evaluation.

---

## 📖 Reference

```bibtex
@inproceedings{liang2023badminton,
  title={Badminton Action Classification Based on Human Skeleton Data Extracted by AlphaPose},
  author={Liang, Zhantu and Nyamasvisva, Tadiwa Elisha},
  booktitle={2023 International Conference on Sensing, Measurement \& Data Analytics (ICSMD)},
  year={2023},
  doi={10.1109/ICSMD60522.2023.10490491}
}
```
