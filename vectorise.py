import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib
from print_data import print_stuff


def vectorise(
    in_path="assets/processed/misinfo-dataset.csv",
    out_X="assets/features/tfidf-features.npz",
    out_y="assets/features/labels.csv",
    out_vec="assets/features/vectoriser.pkl",
):
    # Convert text into TF-IDF features and save artifacts
    df = pd.read_csv(in_path)
    X, y = df["text"], df["label"]

    vectoriser = TfidfVectorizer(stop_words="english")
    X_tfidf = vectoriser.fit_transform(X)

    sp.save_npz(out_X, X_tfidf)
    y.to_csv(out_y, index=False)
    joblib.dump(vectoriser, out_vec)

    print(f"TF-IDF features saved: {out_X}, {out_y}, {out_vec}")


if __name__ == "__main__":
    vectorise()
    print_stuff()
