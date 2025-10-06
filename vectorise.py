#!/usr/bin/env python
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from print_data import print_stuff


def vectorise(
    in_path="./assets/raw/claims.csv",
    out_X="./assets/features/tfidf_features.npz",
    out_y="./assets/features/labels.csv",
    out_vec="./assets/features/vectoriser.pkl",
):
    # Convert text into TF-IDF features and save artifacts
    df = pd.read_csv(in_path)
    X = df["claim"]
    y = df["label"]

    vectoriser = TfidfVectorizer(
        lowercase=True, stop_words="english", max_features=5000, ngram_range=(1, 3)
    )
    X_tfidf = vectoriser.fit_transform(X)

    sp.save_npz(out_X, X_tfidf)
    y.to_csv(out_y, index=False)
    joblib.dump(vectoriser, out_vec)

    print(f"TF-IDF Data Saved: {out_X}, {out_y}, {out_vec}")


if __name__ == "__main__":
    vectorise()
    print_stuff()
