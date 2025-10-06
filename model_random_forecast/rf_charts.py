import os
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils_but_correct_module import load_features

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "rf_model")
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")


def chart_feature_importance(model: Any, vectoriser: Any, top_k: int = 30):
    print("loading feature importance chart")
    # x_train, y_train = load_features()  # training data
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_idx = indices[:top_k]
    top_feats = vectoriser.get_feature_names_out()[top_idx]
    # If vectoriser not loaded yet, load it now
    if top_feats is None:
        top_feats = vectoriser.get_feature_names_out()[top_idx]
    top_imp = importances[top_idx]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_imp[::-1], y=top_feats[::-1], palette="viridis")
    plt.title(f"Top {top_k} TF‑IDF Features by Importance")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()


def chart_confusion_matrix(
    model: Any,
    vectoriser: Any,
    claims_csv_path: str = "./assets/process/claim_test.csv",
):
    print(f"loading claims csv ({claims_csv_path})")
    df_test = pd.read_csv(claims_csv_path)
    x_test = df_test["claim"].astype(str)
    y_test = df_test["label"]

    print("loading confusion matrix")
    x_test_tfidf = vectoriser.transform(x_test)
    y_pred = model.predict(x_test_tfidf)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Non‑Misinfo", "Misinfo"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.ax_.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()


def chart_learning_curve(model: Any, vectoriser: Any):
    print("loading features for training")
    x_train, y_train = load_features()

    print("testing model on training data")
    curve = learning_curve(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    train_sizes, train_scores, val_scores = (curve[0], curve[1], curve[2])

    print("calculating results")
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    print("loading learning curve chart")
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training F1")
    plt.plot(train_sizes, val_mean, "o-", color="orange", label="Cross‑Val F1")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color="orange",
    )
    plt.title("Learning Curve (F1 Score)")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def chart(model_name: str, chart_name: str = "all"):
    """
    Load the specified model and plot one or more diagnostics.

    Parameters
    ----------
    model_name : str
        File name (without .pkl) of the model stored in MODELS_DIR.
    chart_name : str, optional
        Which chart to produce.  Acceptable values:
          * 'feature_importances'
          * 'confusion_matrix'
          * 'learning_curve'
          * 'all'   (default – plot all three)
    """
    print("loading model")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    model = joblib.load(model_path)

    print("loading vectoriser")
    vectoriser = joblib.load("./assets/features/vectoriser.pkl")

    if chart_name in ("feature_importances"):
        chart_feature_importance(model, vectoriser, 30)
    if chart_name in ("confusion_matrix"):
        chart_confusion_matrix(model, vectoriser)
    if chart_name in ("learning_curve", "all"):
        chart_learning_curve(model, vectoriser)
