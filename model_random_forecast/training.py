import os
import time

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


def train(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    print("loading features")
    x_train, y_train = load_features()

    print("fitting data into RandomForecastClassifier")
    model = RandomForestClassifier(n_estimators=700, max_depth=20, min_samples_split=4, min_samples_leaf=2)
    model.fit(x_train, y_train)

    print(f"saving model to: {model_path}")
    joblib.dump(model, model_path)


def chart_tree_count_affect_rsme():
    """
    Train a RandomForest on *only 20 %* of the available data, sweep the
    number of trees from 1 - 1000 (step = 50), record the RMSE on the
    held‑out 80 % and finally plot the curve to determine what the optimal
    tree count is.
    """
    print("loading all features, and only taking 20%")
    x, y = load_features()
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, train_size=0.20, stratify=y, random_state=42
    )

    max_tree_count = 1000
    tree_count_increment = 50
    tree_counts = list(range(50, max_tree_count, tree_count_increment))
    rmse_list = []

    def _train_one(n_trees: int):
        # RandomForest already parallelises bagging internally
        print(f" -> testing n={n_trees} (out of {max_tree_count})...")
        start = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=-1,  # <-- parallelise the *forest itself*
        )
        rf.fit(X_train, y_train)

        probs = rf.predict_proba(X_val)
        prob_pos = np.take(probs, 1, axis=1).ravel()  # probability of class “1”

        rmse = np.sqrt(mean_squared_error(y_val, prob_pos))
        elapsed = time.time() - start
        print(f" -> done {n_trees}/{max_tree_count} (time={elapsed}s rsme={rmse})")

        return n_trees, rmse

    print(
        f"beginning test, up to {max_tree_count} trees, incrementing by {tree_count_increment} each time. this will be done in parallel."
    )

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_train_one)(nt) for nt in tree_counts
    )
    tree_counts, rmse_list = zip(*results)

    print("plotting results")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(tree_counts, rmse_list, marker="o", color="steelblue")
    plt.title("RMSE vs. Number of Trees (Random Forest)")
    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()


def test(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    print(f"loading model from: {model_path}")
    model = joblib.load(model_path)

    print("loading vectors from feature extraction")
    vec_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "features", "vectoriser.pkl"
    )
    vec_path = os.path.abspath(vec_path)
    vectoriser = joblib.load(vec_path)

    print("generating predictions")
    vectoriser = joblib.load("./assets/features/vectoriser.pkl")
    df = pd.read_csv("./assets/process/claim_test.csv")
    x_test = df["claim"].astype(str)
    y_test = df["label"]

    x_test_tfidf = vectoriser.transform(x_test)
    predictions = model.predict(x_test_tfidf)
    correct = (predictions == y_test).sum()
    total = len(y_test)
    score = model.score(x_test_tfidf, y_test)

    print("- - - - - Random Forecast Classifier - - - - - -")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Model Score: {score:.4f}  ({score * 100:.2f}%)")
