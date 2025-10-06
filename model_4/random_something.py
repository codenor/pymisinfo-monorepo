#!/usr/bin/env python
# logreg_supervised.py
# ---------------------------------------------
# Loads the TF‑IDF matrix, labels, vectoriser and the
# RandomForest model trained in `vectorise.py`,
# evaluates on a held‑out test split and prints
# a classification report.
# ---------------------------------------------

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    print(f"saving model to: {model_path}")
    joblib.dump(model, model_path)

    # print(f"saving vectors to: {VECTORISER_PATH}")
    # joblib.dump(load_features.__dict__["vectoriser"], VECTORISER_PATH)


def test(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    print(f"loading model from: {model_path}")
    model = joblib.load(model_path)

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


def list_models():
    dir = os.listdir(MODELS_DIR)
    for d in dir:
        print(d)


def help():
    print("./random_something.py <command>")
    print("possible commands: [ train <model-name>, test <model-name>, list ]")
    sys.exit(-1)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        help()

    match sys.argv[1]:
        case "train": 
            train(sys.argv[2])
        case "test":
            test(sys.argv[2])
        case "list":
            list_models()
        case _:
            help()
