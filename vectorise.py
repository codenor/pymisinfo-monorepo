#!/usr/bin/env python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib
from print_data import print_stuff


def vectorise(
    in_path="./assets/processed/misinfo_dataset.csv",
    out_X="./assets/features/tfidf_features.npz",
    out_y="./assets/features/labels.csv",
    out_vec="./assets/features/vectoriser.pkl",
):
    df = pd.read_csv(in_path)
    X = df["claim"]
    y = df["label"]

    print(
        f"Starting TF-IDF vectorisation on {len(X):,} claims. . . (This may take awhile)"
    )
    vectoriser = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 3)
    )
    X_tfidf = vectoriser.fit_transform(X)
    print("TF-IDF vectorisation complete.")
    print("Please wait while files are being saved. . .")

    sp.save_npz(out_X, X_tfidf)
    y.to_csv(out_y, index=False)
    joblib.dump(vectoriser, out_vec)

    print(f"TF-IDF Data Saved: {out_X}, {out_y}, {out_vec}")


if __name__ == "__main__":
    vectorise()
    print_stuff()
