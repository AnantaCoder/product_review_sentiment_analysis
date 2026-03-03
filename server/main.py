import os
import pickle
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Bidirectional,
    Dropout,
    SpatialDropout1D,
    BatchNormalization,
    Layer,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str


class SelfAttention(Layer):
    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(feat_dim, feat_dim), initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(shape=(feat_dim,), initializer="zeros", trainable=True)
        self.u = self.add_weight(
            shape=(feat_dim, 1), initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        v = tf.tanh(tf.matmul(x, self.W) + self.b)
        scores = tf.matmul(v, self.u)
        alpha = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context


SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SERVER_DIR)
MODEL_PATH = os.path.join(
    ROOT_DIR, "deep_learning", "trained_models", "best_sentiment_model.pkl"
)

print(f"Loading model from: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

VOCAB_SIZE = bundle["vocab_size"]
MAX_LEN = bundle["max_length"]
EMBEDDING_DIM = bundle["embedding_dim"]
tokenizer = bundle["tokenizer"]
best_thresholds = np.array(bundle["best_thresholds"])
class_map = bundle["class_map"]


inputs = Input(shape=(MAX_LEN,))
x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN)(inputs)
x = SpatialDropout1D(0.3)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(
    x
)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(
    x
)
x = SelfAttention()(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(3, activation="softmax")(x)

model = Model(inputs, outputs)
model.set_weights(bundle["weights"])

print("Model loaded and ready.")


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    review_text = data.review

    seq = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(padded)[0]
    adjusted = probs / best_thresholds
    predicted_class = int(np.argmax(adjusted))

    sentiment = class_map[predicted_class]
    confidence = float(probs[predicted_class])

    # Generate insights
    word_count = len(review_text.split())

    class_probabilities = {
        class_map[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_map))
    }

    adjusted_scores = {
        class_map[i]: round(float(adjusted[i]), 4) for i in range(len(class_map))
    }

    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "insights": {
            "word_count": word_count,
            "raw_probabilities_percent": class_probabilities,
            "calibrated_scores": adjusted_scores,
        },
    }
