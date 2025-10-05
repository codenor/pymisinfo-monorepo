import os
import joblib
import pandas as pd
import scipy.sparse as sp
from utils import load_features
from sklearn.linear_model import LogisticRegression


def run_logreg():
    # Load TF-IDF features + labels
    x_train, y_train = load_features()
    
    # Train logistic regression
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)

    # Build absolute path to vectoriser.pkl
    vec_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "features", "vectoriser.pkl"
        )
    vec_path = os.path.abspath(vec_path)

    # Load vectorizer
    vectorizer = joblib.load(vec_path)

    # Transform and predict test data
    vectorizer = joblib.load("./assets/features/vectoriser.pkl")
    df = pd.read_csv("./assets/process/claim_test.csv")
    x_test = df["claim"].astype(str)
    y_test = df["label"]

    x_test_tfidf = vectorizer.transform(x_test)
    predictions = model.predict(x_test_tfidf)
    correct = (predictions == y_test).sum()
    total = len(y_test)
    score = model.score(x_test_tfidf, y_test)


    print("- - - - - Logistic Regression - - - - - -")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Model Score: {score:.4f}  ({score * 100:.2f}%)")

if __name__ == "__main__":
    run_logreg()
