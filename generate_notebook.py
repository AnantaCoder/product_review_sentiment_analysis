import json
import os

os.makedirs("models", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🛒 Product Review Sentiment Analysis using Deep Learning\n",
    "\n",
    "**Objective:** Build a deep learning model to classify Amazon product reviews as **Positive** or **Negative**.\n",
    "\n",
    "**Key Challenge:** The dataset is **highly imbalanced** — there are far more positive reviews than negative ones.\n",
    "\n",
    "**Approach:**\n",
    "- Proper EDA with rich visualizations\n",
    "- Text preprocessing & tokenization\n",
    "- SMOTE-like oversampling of minority class + class weights\n",
    "- Bidirectional LSTM with tuned architecture\n",
    "- Comprehensive evaluation with confusion matrix, ROC curve, and per-class analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== IMPORTS ==========\n",
    "import os, re, pickle, warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (classification_report, confusion_matrix,\n",
    "                             roc_curve, auc, precision_recall_curve)\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import (Embedding, LSTM, Dense, Bidirectional,\n",
    "                                     Dropout, SpatialDropout1D, GlobalMaxPooling1D,\n",
    "                                     BatchNormalization, Conv1D, MaxPooling1D)\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)\n",
    "print('TensorFlow version:', tf.__version__)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📊 1. Data Loading & Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../datasets/amazon.csv')\n",
    "print(f'Dataset shape: {df.shape}')\n",
    "print(f'\\nColumns: {list(df.columns)}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Missing values:')\n",
    "print(df.isnull().sum())\n",
    "print(f'\\nTotal duplicate rows: {df.duplicated().sum()}')\n",
    "print(f'\\nRating distribution:\\n{df[\"overall\"].value_counts().sort_index()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 EDA Chart 1: Rating Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Bar chart\n",
    "colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']\n",
    "rating_counts = df['overall'].value_counts().sort_index()\n",
    "axes[0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', linewidth=0.5)\n",
    "axes[0].set_xlabel('Rating (Stars)', fontsize=12)\n",
    "axes[0].set_ylabel('Number of Reviews', fontsize=12)\n",
    "axes[0].set_title('Distribution of Star Ratings', fontsize=14, fontweight='bold')\n",
    "for i, (idx, val) in enumerate(zip(rating_counts.index, rating_counts.values)):\n",
    "    axes[0].text(idx, val + 20, str(val), ha='center', fontweight='bold', fontsize=10)\n",
    "\n",
    "# Pie chart\n",
    "axes[1].pie(rating_counts.values, labels=[f'{i}⭐' for i in rating_counts.index],\n",
    "            autopct='%1.1f%%', colors=colors, startangle=140,\n",
    "            explode=[0.05]*len(rating_counts), shadow=True, textprops={'fontsize': 11})\n",
    "axes[1].set_title('Rating Proportions', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_rating_distribution.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🏷️ Sentiment Labeling\n",
    "Map ratings to binary sentiment: **Positive** (4-5) vs **Negative** (1-2). Remove neutral (3) reviews."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna(subset=['reviewText', 'overall'])\n",
    "df = df[df['overall'] != 3].copy()\n",
    "df['sentiment'] = df['overall'].apply(lambda x: 1 if x > 3 else 0)\n",
    "df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})\n",
    "\n",
    "print(f'After filtering: {len(df)} reviews')\n",
    "print(f'\\nSentiment distribution:\\n{df[\"sentiment\"].value_counts()}')\n",
    "print(f'\\nImbalance ratio: {df[\"sentiment\"].value_counts()[1] / df[\"sentiment\"].value_counts()[0]:.1f}x more positive')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 EDA Chart 2: Sentiment Class Imbalance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
    "\n",
    "sent_counts = df['sentiment_label'].value_counts()\n",
    "bar_colors = ['#e74c3c', '#2ecc71']\n",
    "\n",
    "# Bar chart\n",
    "bars = axes[0].bar(sent_counts.index, sent_counts.values, color=bar_colors,\n",
    "                   edgecolor='black', linewidth=0.5, width=0.5)\n",
    "for bar, val in zip(bars, sent_counts.values):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,\n",
    "                 f'{val}\\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')\n",
    "axes[0].set_ylabel('Count', fontsize=12)\n",
    "axes[0].set_title('⚠️ Severe Class Imbalance', fontsize=14, fontweight='bold', color='darkred')\n",
    "\n",
    "# Donut chart\n",
    "wedges, texts, autotexts = axes[1].pie(sent_counts.values, labels=sent_counts.index,\n",
    "    autopct='%1.1f%%', colors=bar_colors, startangle=90, pctdistance=0.75,\n",
    "    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2), textprops={'fontsize':12})\n",
    "axes[1].set_title('Sentiment Split', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_class_imbalance.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 EDA Chart 3: Review Length Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['review_length'] = df['reviewText'].apply(lambda x: len(str(x).split()))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Histogram by sentiment\n",
    "for label, color in zip(['Negative', 'Positive'], ['#e74c3c', '#2ecc71']):\n",
    "    subset = df[df['sentiment_label'] == label]['review_length']\n",
    "    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label, edgecolor='white')\n",
    "axes[0].set_xlabel('Number of Words', fontsize=12)\n",
    "axes[0].set_ylabel('Frequency', fontsize=12)\n",
    "axes[0].set_title('Review Length Distribution by Sentiment', fontsize=14, fontweight='bold')\n",
    "axes[0].legend(fontsize=11)\n",
    "axes[0].set_xlim(0, 500)\n",
    "\n",
    "# Box plot\n",
    "sns.boxplot(data=df, x='sentiment_label', y='review_length', palette=['#e74c3c', '#2ecc71'], ax=axes[1])\n",
    "axes[1].set_ylim(0, 500)\n",
    "axes[1].set_title('Review Length Box Plot', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Sentiment', fontsize=12)\n",
    "axes[1].set_ylabel('Word Count', fontsize=12)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_review_lengths.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print('Average review length:')\n",
    "print(df.groupby('sentiment_label')['review_length'].describe()[['mean', '50%', 'max']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🧹 2. Text Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords', quiet=True)\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "stop_words = set(stopwords.words('english'))\n",
    "# Keep negation words — crucial for sentiment!\n",
    "negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody',\n",
    "                  'nothing', 'nowhere', 'hardly', 'barely', 'scarcely',\n",
    "                  \"don't\", \"doesn't\", \"didn't\", \"won't\", \"wouldn't\",\n",
    "                  \"can't\", \"couldn't\", \"shouldn't\", \"isn't\", \"aren't\",\n",
    "                  \"wasn't\", \"weren't\", \"hasn't\", \"haven't\", \"hadn't\"}\n",
    "stop_words = stop_words - negation_words\n",
    "\n",
    "def clean_text(text):\n",
    "    text = str(text).lower()\n",
    "    text = re.sub(r'<[^>]+>', '', text)           # Remove HTML tags\n",
    "    text = re.sub(r'http\\S+|www\\S+', '', text)    # Remove URLs\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)          # Keep only letters\n",
    "    tokens = text.split()\n",
    "    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]\n",
    "    return ' '.join(tokens)\n",
    "\n",
    "df['clean_text'] = df['reviewText'].apply(clean_text)\n",
    "print('Sample cleaned reviews:')\n",
    "for i in range(3):\n",
    "    print(f'  [{df.iloc[i][\"sentiment_label\"]}] {df.iloc[i][\"clean_text\"][:100]}...')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ✂️ 3. Train-Test Split & Tokenization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df['clean_text'].values\n",
    "y = df['sentiment'].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "print(f'Train: {len(X_train)} | Test: {len(X_test)}')\n",
    "print(f'Train class distribution: {Counter(y_train)}')\n",
    "print(f'Test  class distribution: {Counter(y_test)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "VOCAB_SIZE = 15000\n",
    "MAX_LEN = 200\n",
    "EMBEDDING_DIM = 128\n",
    "\n",
    "tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')\n",
    "tokenizer.fit_on_texts(X_train)\n",
    "\n",
    "X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train),\n",
    "                            maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test),\n",
    "                           maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "\n",
    "print(f'Vocabulary size: {min(len(tokenizer.word_index)+1, VOCAB_SIZE)}')\n",
    "print(f'Train shape: {X_train_seq.shape}')\n",
    "print(f'Test  shape: {X_test_seq.shape}')\n",
    "\n",
    "# Save tokenizer\n",
    "with open('../trained_models/tokenizer.pkl', 'wb') as f:\n",
    "    pickle.dump(tokenizer, f)\n",
    "print('\\n✅ Tokenizer saved to trained_models/tokenizer.pkl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ⚖️ 4. Handling Class Imbalance\n",
    "\n",
    "We use **two strategies** to combat the severe imbalance:\n",
    "1. **Random oversampling** of the minority class (negative reviews) to balance the training data\n",
    "2. **Class weights** as a secondary safeguard during training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Strategy 1: Oversample minority class ---\n",
    "neg_indices = np.where(y_train == 0)[0]\n",
    "pos_indices = np.where(y_train == 1)[0]\n",
    "\n",
    "n_pos = len(pos_indices)\n",
    "n_neg = len(neg_indices)\n",
    "print(f'Before oversampling — Positive: {n_pos}, Negative: {n_neg}')\n",
    "\n",
    "# Oversample negative reviews to match positive count\n",
    "oversampled_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=True)\n",
    "balanced_indices = np.concatenate([pos_indices, oversampled_neg_indices])\n",
    "np.random.shuffle(balanced_indices)\n",
    "\n",
    "X_train_balanced = X_train_seq[balanced_indices]\n",
    "y_train_balanced = y_train[balanced_indices]\n",
    "\n",
    "print(f'After  oversampling — Positive: {sum(y_train_balanced==1)}, Negative: {sum(y_train_balanced==0)}')\n",
    "print(f'Total training samples: {len(y_train_balanced)}')\n",
    "\n",
    "# --- Strategy 2: Class weights (lighter, as backup) ---\n",
    "classes = np.unique(y_train_balanced)\n",
    "weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)\n",
    "class_weights = dict(zip(classes, weights))\n",
    "print(f'\\nClass weights: {class_weights}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 EDA Chart 4: Before vs After Balancing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "before = [n_neg, n_pos]\n",
    "after = [sum(y_train_balanced==0), sum(y_train_balanced==1)]\n",
    "labels = ['Negative', 'Positive']\n",
    "bar_colors = ['#e74c3c', '#2ecc71']\n",
    "\n",
    "for ax, vals, title in zip(axes, [before, after], ['Before Oversampling', 'After Oversampling']):\n",
    "    bars = ax.bar(labels, vals, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.5)\n",
    "    for bar, v in zip(bars, vals):\n",
    "        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,\n",
    "                str(v), ha='center', fontweight='bold', fontsize=11)\n",
    "    ax.set_title(title, fontsize=13, fontweight='bold')\n",
    "    ax.set_ylabel('Count')\n",
    "\n",
    "plt.suptitle('Training Data Balancing via Oversampling', fontsize=15, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_oversampling.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🧠 5. Model Architecture\n",
    "\n",
    "**Improved architecture:**\n",
    "- Larger embedding dimension (128)\n",
    "- `SpatialDropout1D` after embedding (regularizes word vectors)\n",
    "- Stacked Bidirectional LSTM layers\n",
    "- Batch normalization for training stability\n",
    "- Learning rate scheduler for fine-tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential([\n",
    "    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),\n",
    "    SpatialDropout1D(0.3),\n",
    "    \n",
    "    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),\n",
    "    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),\n",
    "    \n",
    "    BatchNormalization(),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dropout(0.4),\n",
    "    Dense(32, activation='relu'),\n",
    "    Dropout(0.3),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "optimizer = Adam(learning_rate=0.001)\n",
    "model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🏋️ 6. Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "EPOCHS = 15\n",
    "BATCH_SIZE = 64\n",
    "\n",
    "callbacks = [\n",
    "    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),\n",
    "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)\n",
    "]\n",
    "\n",
    "history = model.fit(\n",
    "    X_train_balanced, y_train_balanced,\n",
    "    epochs=EPOCHS,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    validation_data=(X_test_seq, y_test),\n",
    "    class_weight=class_weights,\n",
    "    callbacks=callbacks,\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📊 7. Training History Charts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Loss\n",
    "axes[0].plot(history.history['loss'], 'o-', label='Train Loss', color='#3498db', linewidth=2)\n",
    "axes[0].plot(history.history['val_loss'], 's-', label='Val Loss', color='#e74c3c', linewidth=2)\n",
    "axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend(fontsize=11)\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Accuracy\n",
    "axes[1].plot(history.history['accuracy'], 'o-', label='Train Accuracy', color='#2ecc71', linewidth=2)\n",
    "axes[1].plot(history.history['val_accuracy'], 's-', label='Val Accuracy', color='#e67e22', linewidth=2)\n",
    "axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('Accuracy')\n",
    "axes[1].legend(fontsize=11)\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/training_history.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📋 8. Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict\n",
    "y_pred_prob = model.predict(X_test_seq).flatten()\n",
    "y_pred = (y_pred_prob > 0.5).astype(int)\n",
    "\n",
    "print('=' * 55)\n",
    "print('           CLASSIFICATION REPORT')\n",
    "print('=' * 55)\n",
    "print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 Evaluation Chart 1: Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Raw counts\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],\n",
    "            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],\n",
    "            annot_kws={'size': 16})\n",
    "axes[0].set_xlabel('Predicted', fontsize=12)\n",
    "axes[0].set_ylabel('Actual', fontsize=12)\n",
    "axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')\n",
    "\n",
    "# Normalized\n",
    "cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges', ax=axes[1],\n",
    "            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],\n",
    "            annot_kws={'size': 16})\n",
    "axes[1].set_xlabel('Predicted', fontsize=12)\n",
    "axes[1].set_ylabel('Actual', fontsize=12)\n",
    "axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/confusion_matrix.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 Evaluation Chart 2: ROC Curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, _ = roc_curve(y_test, y_pred_prob)\n",
    "roc_auc = auc(fpr, tpr)\n",
    "\n",
    "plt.figure(figsize=(7, 6))\n",
    "plt.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')\n",
    "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')\n",
    "plt.fill_between(fpr, tpr, alpha=0.15, color='#3498db')\n",
    "plt.xlabel('False Positive Rate', fontsize=12)\n",
    "plt.ylabel('True Positive Rate', fontsize=12)\n",
    "plt.title('ROC Curve', fontsize=14, fontweight='bold')\n",
    "plt.legend(loc='lower right', fontsize=11)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/roc_curve.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 Evaluation Chart 3: Precision-Recall Curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)\n",
    "\n",
    "plt.figure(figsize=(7, 6))\n",
    "plt.plot(recall_vals, precision_vals, color='#e67e22', linewidth=2.5, label='Precision-Recall Curve')\n",
    "plt.fill_between(recall_vals, precision_vals, alpha=0.15, color='#e67e22')\n",
    "plt.xlabel('Recall', fontsize=12)\n",
    "plt.ylabel('Precision', fontsize=12)\n",
    "plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')\n",
    "plt.legend(fontsize=11)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/precision_recall_curve.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📈 Evaluation Chart 4: Per-Class Performance Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)\n",
    "\n",
    "metrics = ['precision', 'recall', 'f1-score']\n",
    "neg_scores = [report['Negative'][m] for m in metrics]\n",
    "pos_scores = [report['Positive'][m] for m in metrics]\n",
    "\n",
    "x = np.arange(len(metrics))\n",
    "width = 0.3\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "bars1 = ax.bar(x - width/2, neg_scores, width, label='Negative', color='#e74c3c', edgecolor='black', linewidth=0.5)\n",
    "bars2 = ax.bar(x + width/2, pos_scores, width, label='Positive', color='#2ecc71', edgecolor='black', linewidth=0.5)\n",
    "\n",
    "# Add value labels\n",
    "for bars in [bars1, bars2]:\n",
    "    for bar in bars:\n",
    "        height = bar.get_height()\n",
    "        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n",
    "                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)\n",
    "\n",
    "ax.set_ylabel('Score', fontsize=12)\n",
    "ax.set_title('Per-Class Model Performance', fontsize=14, fontweight='bold')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=12)\n",
    "ax.legend(fontsize=11)\n",
    "ax.set_ylim(0, 1.15)\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/per_class_performance.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 💾 9. Save Best Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save as .h5 (native Keras format)\n",
    "model.save('../trained_models/best_sentiment_model.h5')\n",
    "print('✅ Model saved as best_sentiment_model.h5')\n",
    "\n",
    "# Save as .pkl (model config + weights — as requested)\n",
    "model_data = {\n",
    "    'config': model.get_config(),\n",
    "    'weights': model.get_weights(),\n",
    "    'vocab_size': VOCAB_SIZE,\n",
    "    'max_length': MAX_LEN,\n",
    "    'embedding_dim': EMBEDDING_DIM\n",
    "}\n",
    "with open('../trained_models/best_sentiment_model.pkl', 'wb') as f:\n",
    "    pickle.dump(model_data, f)\n",
    "print('✅ Model saved as best_sentiment_model.pkl')\n",
    "\n",
    "print('\\n📁 Files in trained_models:')\n",
    "for fname in os.listdir('../trained_models'):\n",
    "    fsize = os.path.getsize(f'../trained_models/{fname}') / (1024*1024)\n",
    "    print(f'  {fname} ({fsize:.2f} MB)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🔮 10. Test Predictions on Sample Reviews"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_reviews = [\n",
    "    'This product is amazing! Best purchase I have ever made. Highly recommended!',\n",
    "    'Terrible quality. Broke after one week. Complete waste of money.',\n",
    "    'Works as expected, nothing special but does the job.',\n",
    "    'Absolutely horrible experience. The item arrived damaged and customer service was unhelpful.',\n",
    "    'Love it! Great value for money and super fast shipping.'\n",
    "]\n",
    "\n",
    "cleaned = [clean_text(r) for r in sample_reviews]\n",
    "seq = pad_sequences(tokenizer.texts_to_sequences(cleaned), maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "preds = model.predict(seq).flatten()\n",
    "\n",
    "print('\\n' + '='*65)\n",
    "print('  SAMPLE PREDICTIONS')\n",
    "print('='*65)\n",
    "for review, prob in zip(sample_reviews, preds):\n",
    "    sentiment = '✅ POSITIVE' if prob > 0.5 else '❌ NEGATIVE'\n",
    "    print(f'\\n  Review: \\\"{review[:70]}...\\\"')\n",
    "    print(f'  Prediction: {sentiment} (confidence: {max(prob, 1-prob)*100:.1f}%)')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("models/sentiment_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("✅ Notebook generated successfully at models/sentiment_analysis.ipynb")
