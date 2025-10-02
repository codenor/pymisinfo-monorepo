import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    silhouette_score,
)


def load_features():
    x = sp.load_npz("/data/features/tfidf_features.npz")
    y = pd.read_csv("/data/features/labels.csv")["label"]
    return x, y


def train_test_split_data(x, y, test_size=0.2, stratify=True):
    return train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y if stratify else None
    )


def evaluate_supervised(model, X_test, y_test):
    preds = model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


def evaluate_unsupervised(model, X, y):
    clusters = model.fit_predict(X)
    print("ARI (vs labels):", adjusted_rand_score(y, clusters))
    print("Silhouette:", silhouette_score(X, clusters))
