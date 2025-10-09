#!/usr/bin/env python
import os
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def vectorise(
    in_path="./assets/processed/misinfo_dataset.csv",
    out_dir="./assets/features/",
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading dataset from {in_path}")
    df = pd.read_csv(in_path)
    X_text = df["claim"].astype(str)
    y = df["label"]

    # Split BEFORE vectorisation (70/15/15)
    print("Splitting dataset: 70% train, 15% validation, 15% test...")
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(
        f"Training: {len(X_train_text):,} | Validation: {len(X_val_text):,} | Test: {len(X_test_text):,}"
    )

    # Fit TF-IDF ONLY on training data
    print("Fitting TF-IDF vectoriser on training data only...")
    vectoriser = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
    )
    X_train_tfidf = vectoriser.fit_transform(X_train_text)

    # Transform validation/test using same vectoriser
    X_val_tfidf = vectoriser.transform(X_val_text)
    X_test_tfidf = vectoriser.transform(X_test_text)

    # Save all splits + vectoriser
    print("Saving TF-IDF matrices and labels...")
    sp.save_npz(os.path.join(out_dir, "X_train_tfidf.npz"), X_train_tfidf)
    sp.save_npz(os.path.join(out_dir, "X_val_tfidf.npz"), X_val_tfidf)
    sp.save_npz(os.path.join(out_dir, "X_test_tfidf.npz"), X_test_tfidf)

    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(out_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    joblib.dump(vectoriser, os.path.join(out_dir, "vectoriser.pkl"))

    print(f"TF-IDF data saved in {out_dir}")
    print("Files generated:")
    print(" - X_train_tfidf.npz")
    print(" - X_val_tfidf.npz")
    print(" - X_test_tfidf.npz")
    print(" - y_train.csv / y_val.csv / y_test.csv")
    print(" - vectoriser.pkl")


if __name__ == "__main__":
    vectorise()
