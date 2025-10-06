import os
import joblib
import pandas as pd
import scipy.sparse as sp
from utils import load_features
from sklearn.linear_model import LogisticRegression


def run_logreg():
    model_path = "./assets/logreg_model.pkl"
    vec_path = "./assets/features/vectoriser.pkl"
    data_path = "./assets/processed/misinfo_dataset.csv"

    # Load or train model
    if os.path.exists(model_path):
        print("Loading existing Logistic Regression model...")
        model = joblib.load(model_path)
    else:
        print("Training new Logistic Regression model...")
        x_train, y_train = load_features()
        model = LogisticRegression(max_iter=200)
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    # Load vectorizer
    vectorizer = joblib.load(vec_path)

    # Load dataset for evaluation
    df = pd.read_csv(data_path)
    x_test = df["claim"].astype(str)
    y_test = df["label"]

    x_test_tfidf = vectorizer.transform(x_test)
    predictions = model.predict(x_test_tfidf)
    correct = (predictions == y_test).sum()
    total = len(y_test)
    score = model.score(x_test_tfidf, y_test)

    print("- - - - - Logistic Regression - - - - - -")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Model Score: {score:.4f} ({score * 100:.2f}%)")


if __name__ == "__main__":
    run_logreg()
