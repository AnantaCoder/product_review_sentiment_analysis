import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.makedirs('trained_models', exist_ok=True)

print("Loading Data...")
data_path = 'datasets/amazon.csv'
df = pd.read_csv(data_path)

df = df.dropna(subset=['reviewText', 'overall'])
df = df[df['overall'] != 3]
df['sentiment'] = df['overall'].apply(lambda x: 1 if x > 3 else 0)

print("Data cleaning...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean_text'] = df['reviewText'].apply(clean_text)

X = df['clean_text'].values
y = df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Tokenizing...")
vocab_size = 10000
max_length = 150
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length, padding='post', truncating='post')

with open('trained_models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Computing class weights...")
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("Building Model...")
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training Model...")
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_pad, y_test),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

print("Evaluating...")
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred))

print("Saving Models...")
model.save('trained_models/best_model.h5')
# Save weights to pkl as well
model_data = {'config': model.get_config(), 'weights': model.get_weights()}
with open('trained_models/best_model_weights.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("Done!")
