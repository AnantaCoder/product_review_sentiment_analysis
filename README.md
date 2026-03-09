# Product Review Sentiment Analysis

### Imbalance-Aware Deep Learning with Bidirectional LSTM + Self-Attention

A full-stack application that classifies Amazon product reviews into **Negative**, **Neutral**, or **Positive** sentiment using a deep learning model explicitly designed to handle severe class imbalance, served through a FastAPI backend and a Next.js frontend.

---

## Project Structure

```
├── deep_learning/           # Model training notebook, dataset, and saved models
│   ├── datasets/            # Amazon product review CSV
│   ├── models/              # Jupyter notebook (training pipeline) + output charts
│   └── trained_models/      # Exported model weights (.h5 / .pkl)
├── server/                  # FastAPI backend serving predictions
│   └── main.py
├── frontend/                # Next.js + React + Tailwind CSS UI
│   └── src/
│       ├── app/             # Pages & layout
│       └── components/      # AnalyzeTab, HistoryTab, Header
└── env/                     # Python virtual environment
```

---

## Tech Stack

| Layer         | Technology                                                        |
| ------------- | ----------------------------------------------------------------- |
| Deep Learning | TensorFlow / Keras (BiLSTM + Self-Attention)                      |
| Backend API   | FastAPI, Uvicorn, Pydantic                                        |
| Frontend      | Next.js 16, React 19, Tailwind CSS 4, TypeScript                  |
| Data Science  | Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud |

---

## Dataset

| Property            | Value                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------- |
| **Source**          | Amazon product reviews (`deep_learning/datasets/amazon.csv`)                                 |
| **Key columns**     | `reviewText` (raw text), `overall` (star rating 1–5)                                         |
| **Label mapping**   | Rating > 3 → **Positive (2)** · Rating = 3 → **Neutral (1)** · Rating < 3 → **Negative (0)** |
| **Class imbalance** | Positive >> Neutral >> Negative (typical ratio ~10 : 1 or worse)                             |

---

## Model Architecture

```
Input (MAX_LEN = 150 token IDs)
│
├─ Embedding (30,000 × 128)
├─ SpatialDropout1D (p = 0.30)
├─ BiLSTM-128 (return_sequences=True, dropout=0.20, recurrent_dropout=0.10)
├─ BiLSTM-64  (return_sequences=True, dropout=0.20, recurrent_dropout=0.10)
├─ Self-Attention (learned per-token importance scores)
├─ BatchNormalization
├─ Dense-64 (ReLU) → Dropout (0.40) → Dense-32 (ReLU)
└─ Dense-3 (Softmax)  →  [P(Negative), P(Neutral), P(Positive)]
```

| Hyperparameter  | Value                                           |
| --------------- | ----------------------------------------------- |
| `VOCAB_SIZE`    | 30,000                                          |
| `MAX_LEN`       | 150                                             |
| `EMBEDDING_DIM` | 128                                             |
| `BATCH_SIZE`    | 64                                              |
| `EPOCHS`        | 10 (early stopping at ~5–8)                     |
| Optimizer       | Adam (lr = 0.0005)                              |
| Loss            | Focal Loss (γ = 2, inverse-frequency α weights) |

---

## Imbalance Handling — 4-Layer Strategy

| Layer | Technique                                       | Stage            |
| ----- | ----------------------------------------------- | ---------------- |
| 1     | Stratified train / val / test splits            | Data splitting   |
| 2     | Partial random oversampling (70 % of majority)  | Data preparation |
| 3     | Focal Loss with inverse-frequency alpha weights | Model training   |
| 4     | Per-class decision threshold calibration        | Inference        |

---

## Text Preprocessing

Each review passes through a deterministic `clean_text()` pipeline:

- Lowercasing → HTML & URL removal → non-alphabetic removal
- NLTK English stop-word removal (**negation words preserved**: _not_, _never_, _don't_, etc.)
- Single-character removal → whitespace normalisation

---

## Training & Evaluation

**Callbacks:** EarlyStopping (patience=4), ReduceLROnPlateau (factor=0.5, patience=2), ModelCheckpoint (save best only)

**Metrics used:**

| Metric                        | Why It Matters                                           |
| ----------------------------- | -------------------------------------------------------- |
| Balanced Accuracy             | Average per-class recall; not inflated by majority class |
| Macro F1                      | Penalises poor minority-class performance equally        |
| Matthews Correlation Coef     | Single ±1 score robust to all class frequencies          |
| Confusion Matrix (counts + %) | Visualises per-class error patterns                      |
| ROC curves (one-vs-rest)      | AUC per class                                            |
| Precision-Recall curves       | More informative than ROC under severe imbalance         |

---

## Tools & Libraries

| Category        | Libraries                                |
| --------------- | ---------------------------------------- |
| Deep Learning   | TensorFlow, Keras                        |
| NLP             | NLTK (stopwords, tokenization)           |
| Data Processing | Pandas, NumPy, Scikit-learn              |
| Visualization   | Matplotlib, Seaborn, WordCloud           |
| Backend         | FastAPI, Uvicorn, Pydantic, SQLAlchemy   |
| Frontend        | Next.js, React, Tailwind CSS, TypeScript |
| AI APIs         | Groq                                     |

---

## Quick Setup

```bash
# Backend
pip install -r server/requirements.txt
cd server && uvicorn main:app --reload    # http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev  # http://localhost:3000
```

---

## Authors

Anirban, Aniket, Arnab, Anushka, Ankan
