#!/usr/bin/env python
# logreg_supervised.py
# ---------------------------------------------
# Loads the TF‑IDF matrix, labels, vectoriser and the
# RandomForest model trained in `vectorise.py`,
# evaluates on a held‑out test split and prints
# a classification report.
# ---------------------------------------------

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils_but_correct_module import load_features


def train():
    print("loading features")
    x_train, y_train = load_features()

    print("fitting data into RandomForecastClassifier")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    print("loading vectors from feature extraction")
    vec_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "features", "vectoriser.pkl"
    )
    vec_path = os.path.abspath(vec_path)
    vectorizer = joblib.load(vec_path)

    print("generating predictions")
    vectorizer = joblib.load("./assets/features/vectoriser.pkl")
    df = pd.read_csv("./assets/process/claim_test.csv")
    x_test = df["claim"].astype(str)
    y_test = df["label"]

    x_test_tfidf = vectorizer.transform(x_test)
    predictions = model.predict(x_test_tfidf)
    correct = (predictions == y_test).sum()
    total = len(y_test)
    score = model.score(x_test_tfidf, y_test)

    print("- - - - - Random Forecast Classifier - - - - - -")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Model Score: {score:.4f}  ({score * 100:.2f}%)")


if __name__ == "__main__":
    train()
