import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from joblib import Parallel, delayed
from rf_utils import load_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split


def load_features():
    x = sp.load_npz("./assets/features/tfidf_features.npz")
    y = pd.read_csv("./assets/features/labels.csv")["label"]
    return x, y


def evaluate_supervised(model, X_test, y_test):
    preds = model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


def evaluate_unsupervised(model, X, y):
    clusters = model.fit_predict(X)
    print("ARI (vs labels):", adjusted_rand_score(y, clusters))
    print("Silhouette:", silhouette_score(X, clusters))


def test_model(model, vectoriser, x_test, y_test):
    """
    Tests a given model, using the dataframe as an input. Will print out the following
    information:
     - correct predictions
     - model score (correct / total)
     - accuracy
     - precision
     - recall
     - f1 score

    Usage:
    ```python
    model = joblib.load(model_path)
    vectoriser = joblib.load(vec_path)
    df = pd.read_csv("./assets/process/claim_test.csv")
    x_test = df["claim"].astype(str)
    y_test = df["label"]

    test_model(model, vectoriser, x_test, y_test)
    ```
    """

    x_test_tfidf = vectoriser.transform(x_test)
    predictions = model.predict(x_test_tfidf)

    # Performance Metrics
    correct = (predictions == y_test).sum()
    total = len(y_test)
    score = model.score(x_test_tfidf, y_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="binary"
    )  # Use 'micro', 'macro' or 'weighted' for multi-class
    recall = recall_score(y_test, predictions, average="binary")
    f1 = f1_score(y_test, predictions, average="binary")
    report = classification_report(y_test, predictions)

    print("- - - - - Random Forest Classifier - - - - - -")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Model Score: {score:.4f}  ({score * 100:.2f}%)")
    print(f"Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"Recall: {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"F1 Score: {f1:.4f}  ({f1 * 100:.2f}%)")
    print("\nClassification Report:\n", report)
