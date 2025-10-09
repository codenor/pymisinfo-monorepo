import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle
from umap import UMAP


def visualise_tfidf_umap_finalzoom(
    x_path="./assets/features/X_train_tfidf.npz",
    y_path="./assets/features/y_train.csv",
    sample_size=8000,
    svd_components=100,
):
    # --- Load data ---
    print("Loading TF-IDF features…")
    X = sp.load_npz(x_path)
    y = pd.read_csv(y_path)["label"]
    X, y = shuffle(X, y, random_state=42)

    # Balance both classes equally for clarity
    df_idx = pd.DataFrame({"idx": np.arange(len(y)), "label": y})
    min_class = df_idx["label"].value_counts().min()
    balanced_idx = (
        df_idx.groupby("label")
        .sample(n=min(sample_size // 2, min_class), random_state=42)
        .idx.values
    )
    X = X[balanced_idx]
    y = y.iloc[balanced_idx].reset_index(drop=True)
    print(f"Balanced sample of {len(y):,} documents loaded.")

    os.makedirs("./assets/plots", exist_ok=True)

    # --- Label mapping (Blue = Misinformation, Orange = True) ---
    label_map = {0: "True", 1: "Misinformation"}
    labels = y.map(label_map)
    labels = pd.Categorical(labels, categories=["Misinformation", "True"])

    # --- Normalise and reduce dimensions ---
    print("Normalising and performing SVD + UMAP…")
    X_norm = Normalizer(copy=False).fit_transform(X)
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd = svd.fit_transform(X_norm)

    umap = UMAP(
        n_neighbors=30,
        min_dist=0.25,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    X_umap = umap.fit_transform(X_svd)

    # --- Compute axis ranges with small margin ---
    x_min, x_max = np.percentile(X_umap[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_umap[:, 1], [1, 99])
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # --- Plot ---
    print("Rendering figure…")
    plt.figure(figsize=(10, 8))
    # Density layer
    sns.kdeplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=labels,
        fill=True,
        alpha=0.25,
        levels=10,
        palette=["#4C72B0", "#DD8452"],
    )
    # Scatter points
    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=labels,
        palette=["#4C72B0", "#DD8452"],
        alpha=0.55,
        s=8,
        edgecolor=None,
        legend="full",
    )

    # Axis and title adjustments
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("TF-IDF Feature Space (SVD + UMAP 2D Projection)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # --- Ensure legend shows both labels ---
    handles, labels_text = plt.gca().get_legend_handles_labels()
    unique_labels = ["Misinformation", "True"]
    unique_handles = [
        handles[labels_text.index(l)] for l in unique_labels if l in labels_text
    ]
    plt.legend(unique_handles, unique_labels, title="Label", loc="upper right")

    plt.tight_layout()
    plt.savefig("./assets/plots/tfidf_umap_finalzoom.png", dpi=300)
    plt.close()

    print("Saved: ./assets/plots/tfidf_umap_finalzoom.png")
    print("   (Balanced, labelled, zoomed out slightly)")


if __name__ == "__main__":
    visualise_tfidf_umap_finalzoom()
