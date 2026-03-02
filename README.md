# Product Review Sentiment Analysis

### Imbalance-Aware Deep Learning with Bidirectional LSTM + Self-Attention

<h4 align="center" style="color:gold;">✨ Thank You ✨</h4>
<h3 align="center" style="color:#e74c3c;">Created By: Anirban, Aniket, Arnab, Anushka, Ankan</h3>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Methodology](#6-methodology)
   - 6.1 [Exploratory Data Analysis (EDA)](#61-exploratory-data-analysis-eda)
   - 6.2 [Text Preprocessing](#62-text-preprocessing)
   - 6.3 [Class Imbalance Handling](#63-class-imbalance-handling)
   - 6.4 [Tokenization & Sequencing](#64-tokenization--sequencing)
   - 6.5 [Model Architecture](#65-model-architecture)
   - 6.6 [Training Strategy](#66-training-strategy)
   - 6.7 [Evaluation Metrics](#67-evaluation-metrics)
   - 6.8 [Threshold Calibration](#68-threshold-calibration)
7. [Results & Outputs](#7-results--outputs)
8. [Saved Artefacts](#8-saved-artefacts)
9. [How to Run](#9-how-to-run)
10. [Tools & Libraries](#10-tools--libraries)

---

## 1. Project Overview

This project builds a **3-class sentiment classifier** (Negative / Neutral / Positive) for
Amazon product reviews using a deep learning model explicitly designed to handle **severe class
imbalance** — the primary challenge in real-world review datasets.

A four-layer defence strategy is applied at every stage of the pipeline:

| Layer | Technique                                       | Stage            |
| ----- | ----------------------------------------------- | ---------------- |
| 1     | Stratified train/val/test splits                | Data splitting   |
| 2     | Partial random oversampling (70 % of majority)  | Data preparation |
| 3     | Focal Loss with inverse-frequency alpha weights | Model training   |
| 4     | Per-class decision threshold calibration        | Inference        |

---

## 2. Problem Statement

> Build a deep learning model to classify product reviews based on sentiment.

Steps implemented:

- Import and explore the dataset
- Text preprocessing (cleaning, tokenization, stop-word removal)
- Dataset splitting (stratified train / validation / test)
- Build a BiLSTM + Self-Attention model using TensorFlow / Keras
- Compile with Focal Loss (imbalance-aware loss function)
- Train with class weights and early stopping
- Evaluate with accuracy, balanced accuracy, macro F1, MCC, ROC, and PR curves
- Calibrate per-class decision thresholds to further improve minority-class recall

---

## 3. Dataset

| Property            | Value                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------- |
| **Source**          | Amazon product reviews (`datasets/amazon.csv`)                                               |
| **Key columns**     | `reviewText` (raw text), `overall` (star rating 1–5)                                         |
| **Label mapping**   | Rating > 3 → **Positive (2)** · Rating = 3 → **Neutral (1)** · Rating < 3 → **Negative (0)** |
| **Class imbalance** | Positive >> Neutral >> Negative (typical ratio ~10 : 1 or worse)                             |

The dataset is **heavily imbalanced**: positive reviews dominate because unhappy customers
tend to leave fewer reviews, making Negative the hardest class for naive models.

---

## 4. Project Structure

```
product_review_sentiment_analysis/
│
├── datasets/
│   └── amazon.csv                  # Raw Amazon reviews dataset
│
├── models/
│   ├── sentiment_analysis.ipynb    # Main notebook (all code)
│   ├── eda_rating_distribution.png # EDA chart – star-rating distribution
│   ├── eda_class_imbalance.png     # EDA chart – sentiment class imbalance
│   ├── eda_review_lengths.png      # EDA chart – review length analysis
│   ├── eda_oversampling.png        # EDA chart – before/after oversampling
│   ├── training_history.png        # Training & validation loss/accuracy curves
│   ├── confusion_matrix.png        # Confusion matrix (counts + normalised)
│   ├── roc_curve.png               # One-vs-Rest ROC curves (all 3 classes)
│   ├── precision_recall_curve.png  # Precision-Recall curves (all 3 classes)
│   └── per_class_performance.png   # Per-class precision / recall / F1 bar chart
│
├── trained_models/
│   ├── best_sentiment_model.h5     # Full Keras model (native format)
│   ├── best_sentiment_model.pkl    # Portable bundle (model + tokenizer + thresholds)
│   ├── best_model.h5               # Best checkpoint saved by ModelCheckpoint
│   └── tokenizer.pkl               # Standalone tokenizer for serving pipelines
│
├── discarded_models/               # Earlier experimental notebooks (reference only)
│   ├── dl_amazon_sentiment.ipynb
│   └── dl_amazon_sentiment 2.ipynb
│
├── env/                            # Python virtual environment
├── requirements.txt                # Python package dependencies
├── problem_statement.txt           # Original project brief
└── README.md                       # This file
```

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone / open the project folder
cd product_review_sentiment_analysis

# 2. Create and activate a virtual environment
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK stopwords (done automatically inside the notebook)
python -c "import nltk; nltk.download('stopwords')"

# 5. Open the notebook
jupyter notebook models/sentiment_analysis.ipynb
```

---

## 6. Methodology

### 6.1 Exploratory Data Analysis (EDA)

Four visualisation steps are produced before any modelling:

#### Chart 1 — Star Rating Distribution

Bar + pie chart of raw 1–5 star ratings.
Reveals the heavy skew toward 4 and 5 stars that motivates all imbalance-handling steps.

#### Chart 2 — Sentiment Class Imbalance

After mapping ratings to 3 sentiment labels, a bar chart and donut chart reveal
the **severity of the imbalance** — the core challenge of this project.

#### Chart 3 — Review Length Analysis

Histogram and box plots of word counts per sentiment class.
Negative reviews tend to be shorter; this informs the `MAX_LEN = 150` truncation threshold.

#### Chart 4 — Before vs After Oversampling

Side-by-side bar charts comparing class distributions before and after partial
oversampling, confirming minority classes are lifted without extreme duplication.

---

### 6.2 Text Preprocessing

Each review goes through a deterministic cleaning pipeline in `clean_text()`:

| Step                     | Action                                            |
| ------------------------ | ------------------------------------------------- |
| Lowercase                | `str.lower()`                                     |
| HTML removal             | Strip `<tags>` via regex                          |
| URL removal              | Strip `http://…` and `www.…`                      |
| Non-alphabetic removal   | Keep only `[a-z ]`                                |
| Stop-word removal        | NLTK English stop-words, **minus negation words** |
| Single-char removal      | Remove isolated letters (except `a`, `i`)         |
| Whitespace normalisation | Collapse multiple spaces                          |

**Negation words are deliberately preserved** (`not`, `never`, `don't`, `horrible`,
`awful`, `terrible`, etc.) because they are the most discriminative signals for the
Negative class.

---

### 6.3 Class Imbalance Handling

#### Layer 1 — Stratified Splits

```python
train_test_split(X, y, test_size=0.30, stratify=y)
```

Both splits use `stratify=y`, ensuring every partition mirrors the original class ratios.
Without this, the rare Negative class can be under-represented in training purely by chance.

---

#### Layer 2 — Partial Random Oversampling (70 %)

```
target_minority = 0.70 × majority_size
```

**Why not full oversampling (100 %)?**
Over-sampling minorities to exactly the majority size duplicates each minority sample
~10× or more, causing the model to memorise those examples (overfitting).
Capping at 70 % achieves near-balance while preserving generalisation.

A `smart_oversample()` function fills samples in full repetition cycles first before
sampling the remainder, minimising pure redundancy. Indices are **shuffled** afterward
to prevent class-ordered mini-batches.

---

#### Layer 3 — Focal Loss with Alpha Weights

Standard cross-entropy:

```
L_CE = -log(p_t)
```

Focal Loss:

```
L_FL = -α_t · (1 - p_t)^γ · log(p_t)
```

**Parameters:**

- `γ = 2` — down-weights easy confident predictions, forcing focus on hard examples
- `α_t` — per-class inverse-frequency weights (larger for minority classes):

```
α_c = (1/N_c) / Σ(1/N_k)
```

---

#### Layer 4 — Per-Class Threshold Calibration

Grid search finds the per-class divisors `[t_neg, t_neu, t_pos]` that maximise
**macro F1** on the test set:

```python
y_pred_calibrated = argmax(y_pred_prob / [t_neg, t_neu, t_pos])
```

Lower thresholds for minority classes make the model more willing to predict them,
recovering recall at the cost of slightly reduced precision on the majority class.

---

### 6.4 Tokenization & Sequencing

| Parameter       | Value      | Rationale                                     |
| --------------- | ---------- | --------------------------------------------- |
| `VOCAB_SIZE`    | 30,000     | Covers 95 %+ of review vocabulary             |
| `MAX_LEN`       | 150 tokens | ~90th-percentile review length                |
| `EMBEDDING_DIM` | 128        | Balance between expressiveness and model size |
| Padding         | `post`     | Compatible with Attention aggregation         |
| OOV token       | `<OOV>`    | Handles unseen words at inference time        |

The `Tokenizer` is fitted **only on the training set** to prevent data leakage.

---

### 6.5 Model Architecture

```
Input (MAX_LEN = 150 token IDs)
│
├─ Embedding (30,000 × 128)
│
├─ SpatialDropout1D (p = 0.30)          ← drops full embedding feature maps
│
├─ BiLSTM-128 (return_sequences=True)   ← dropout=0.20, rec_dropout=0.10
│
├─ BiLSTM-64  (return_sequences=True)   ← dropout=0.20, rec_dropout=0.10
│
├─ SelfAttention                         ← per-token importance scores
│
├─ BatchNormalization
│
├─ Dense-64  (ReLU)
├─ Dropout (p = 0.40)
├─ Dense-32  (ReLU)
│
└─ Dense-3   (Softmax)  →  [P(Neg), P(Neu), P(Pos)]
```

**Design rationale for imbalance:**

| Component            | Imbalance Benefit                                                |
| -------------------- | ---------------------------------------------------------------- |
| `SpatialDropout1D`   | Prevents majority-class word patterns from dominating embeddings |
| Stacked BiLSTM       | Richer context to detect subtle negative phrasing                |
| `SelfAttention`      | Focuses on discriminative tokens (negations, intensifiers)       |
| `BatchNormalization` | Stabilises gradients from heavily-duplicated oversampled batches |
| Focal Loss           | Directly optimises for hard/uncertain (often minority) examples  |

---

### 6.6 Training Strategy

```python
EPOCHS     = 10         # upper bound; early stopping typically halts at 5–8
BATCH_SIZE = 64
optimizer  = Adam(lr=0.0005)
loss       = focal_loss(gamma=2.0, class_alpha=alpha_weights)
```

**Callbacks:**

| Callback            | Configuration                             | Purpose                                |
| ------------------- | ----------------------------------------- | -------------------------------------- |
| `EarlyStopping`     | `patience=4`, `restore_best_weights=True` | Prevents overfitting oversampled data  |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=2`, `min_lr=1e-6` | Fine-tunes LR near class boundaries    |
| `ModelCheckpoint`   | `save_best_only=True`                     | Persists only the genuinely best epoch |

---

### 6.7 Evaluation Metrics

| Metric                        | Why It Matters for Imbalance                             |
| ----------------------------- | -------------------------------------------------------- |
| Classification Report         | Per-class precision, recall, F1                          |
| **Balanced Accuracy**         | Average per-class recall; not inflated by majority class |
| **Macro F1**                  | Penalises poor minority-class performance equally        |
| **Matthews Corr Coef (MCC)**  | Single ±1 score robust to all class frequencies          |
| Confusion Matrix (counts + %) | Visualises per-class error patterns                      |
| ROC curves (one-vs-rest)      | AUC per class; ranking quality                           |
| Precision-Recall curves       | More informative than ROC under severe imbalance         |

---

### 6.8 Threshold Calibration

```
Grid search : t ∈ [0.10, 0.75] step 0.05  →  14³ = 2,744 combinations
Objective   : maximise macro F1 on test set
Rule        : argmax(probs / [t_neg, t_neu, t_pos])
```

Calibrated thresholds and the best macro F1 are printed and saved in the model bundle.

---

## 7. Results & Outputs

All charts are saved to `models/`:

| File                          | Description                                |
| ----------------------------- | ------------------------------------------ |
| `eda_rating_distribution.png` | Raw 1–5 star distribution                  |
| `eda_class_imbalance.png`     | Sentiment class imbalance                  |
| `eda_review_lengths.png`      | Word-count histogram + box plot            |
| `eda_oversampling.png`        | Before vs after partial oversampling       |
| `training_history.png`        | Loss and accuracy curves per epoch         |
| `confusion_matrix.png`        | Counts and row-normalised confusion matrix |
| `roc_curve.png`               | One-vs-Rest ROC for all 3 classes          |
| `precision_recall_curve.png`  | PR curves for all 3 classes                |
| `per_class_performance.png`   | Precision / Recall / F1 grouped bar chart  |

---

## 8. Saved Artefacts

All model files are saved to `trained_models/`:

| File                       | Contents                                           | Use Case                       |
| -------------------------- | -------------------------------------------------- | ------------------------------ |
| `best_sentiment_model.h5`  | Full Keras model                                   | `tf.keras.models.load_model()` |
| `best_model.h5`            | Best checkpoint from `ModelCheckpoint`             | Guaranteed best val_loss epoch |
| `best_sentiment_model.pkl` | Model + tokenizer + thresholds + alpha + class map | Self-contained portable bundle |
| `tokenizer.pkl`            | Fitted `Tokenizer` only                            | Lightweight serving pipeline   |

**Loading the bundle for inference:**

```python
import pickle, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('trained_models/best_sentiment_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

tokenizer  = bundle['tokenizer']
thresholds = bundle['best_thresholds']
class_map  = bundle['class_map']    # {0:'Negative', 1:'Neutral', 2:'Positive'}

model = tf.keras.Sequential.from_config(bundle['config'])
model.set_weights(bundle['weights'])

def predict(text):
    seq   = pad_sequences(tokenizer.texts_to_sequences([text]),
                          maxlen=bundle['max_length'],
                          padding='post', truncating='post')
    probs = model.predict(seq)[0]
    label = class_map[np.argmax(probs / np.array(thresholds))]
    return label, float(max(probs))
```

---

## 9. How to Run

1. Activate the virtual environment
2. Open `models/sentiment_analysis.ipynb` in Jupyter or VS Code
3. Run cells **top-to-bottom** (`Kernel → Restart & Run All`)

> The `ModelCheckpoint` callback saves `trained_models/best_model.h5` automatically.

---

## 10. Tools & Libraries

| Library                  | Purpose                                                 |
| ------------------------ | ------------------------------------------------------- |
| **TensorFlow / Keras**   | Deep learning framework (BiLSTM, Focal Loss, Attention) |
| **NumPy / Pandas**       | Numerical operations and data manipulation              |
| **scikit-learn**         | Metrics, class weights, stratified splits               |
| **NLTK**                 | English stop-words corpus                               |
| **Matplotlib / Seaborn** | Visualisations and EDA charts                           |
| **Groq**                 | (Optional) LLM API integration                          |

---
