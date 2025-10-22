#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt

# ==== Default paths ====
DEF_IN = "./assets/processed/misinfo_dataset.csv"
DEF_FEAT = "./assets/features"
DEF_MAXLEN = 100
DEF_THR = 0.5


def load_data_csv(csv):
    df = pd.read_csv(csv)[["claim", "label"]].dropna().reset_index(drop=True)
    return df


def main(csv, feat, maxlen, thr, limit):
    df = load_data_csv(csv)
    test_npz = np.load(os.path.join(feat, "test_data.npz"))
    model = tf.keras.models.load_model(os.path.join(feat, "bilstm_best.keras"))
    tok = joblib.load(os.path.join(feat, "tokenizer.pkl"))

    X_te, y_te = test_npz["X"], test_npz["y"]
    probs = model.predict(X_te, verbose=0).ravel()
    preds = (probs >= thr).astype(int)

    # === Summary metrics ===
    total = len(y_te)
    # CORRECT: y=1 => misinformation, y=0 => factual
    fp = np.sum((y_te == 0) & (preds == 1))  # factual misclassified as misinfo
    fn = np.sum((y_te == 1) & (preds == 0))  # misinfo misclassified as factual
    acc = np.sum(y_te == preds) / total

    print("\n========== Misclassification Summary ==========")
    print(f"Total test samples: {total:,}")
    print(f"False Positives (factual → misinfo): {fp:,} ({fp/total:.2%})")
    print(f"False Negatives (misinfo → factual): {fn:,} ({fn/total:.2%})")
    print(f"Overall accuracy: {acc:.2%}")
    print(f"Decision threshold: {thr:.3f}")
    print("================================================\n")

    # === Identify indices ===
    idx_fn = np.where((y_te == 1) & (preds == 0))[0]  # misinfo predicted as factual
    idx_fp = np.where((y_te == 0) & (preds == 1))[0]  # factual predicted as misinfo

    # === Print top N examples ===
    print("--- False Negatives (misinfo → factual) ---")
    for i in idx_fn[:limit]:
        print(f"[FN | p={probs[i]:.3f}] {df.iloc[i]['claim'][:200]}")

    print("\n--- False Positives (factual → misinfo) ---")
    for i in idx_fp[:limit]:
        print(f"[FP | p={probs[i]:.3f}] {df.iloc[i]['claim'][:200]}")

    # === Save misclassified samples to CSV ===
    out_idx = np.concatenate([idx_fp, idx_fn])
    out_df = pd.DataFrame({
        "claim": df.loc[out_idx, "claim"].values,
        "true_label": y_te[out_idx],
        "pred_label": preds[out_idx],
        "pred_prob": probs[out_idx]
    })
    out_path = os.path.join(feat, "misclassified_examples.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df):,} misclassified samples → {out_path}")

    # === Word-level diagnostics ===
    fn_texts = [df.iloc[i]["claim"] for i in idx_fn]
    fp_texts = [df.iloc[i]["claim"] for i in idx_fp]

    def top_words(texts, topn=15):
        words = Counter(" ".join(texts).lower().split())
        return words.most_common(topn)

    fn_words = top_words(fn_texts)
    fp_words = top_words(fp_texts)

    print("\n--- Top Words in False Negatives (misinfo → factual) ---")
    for w, c in fn_words:
        print(f"{w:<20} {c}")

    print("\n--- Top Words in False Positives (factual → misinfo) ---")
    for w, c in fp_words:
        print(f"{w:<20} {c}")

    # === Save histogram visualization ===
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(probs[y_te == 1], bins=30, alpha=0.6, label="True = Misinformation (1)")
        plt.hist(probs[y_te == 0], bins=30, alpha=0.6, label="True = Factual (0)")
        plt.axvline(thr, color="red", linestyle="--", label=f"Threshold = {thr:.2f}")
        plt.legend()
        plt.title("Prediction Probability Distributions (1 = misinformation)")
        plt.xlabel("Predicted Probability of Being Misinformation (Class 1)")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # save instead of show
        out_path = os.path.join(feat, "probability_histogram.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"\nSaved probability histogram → {out_path}")
    except Exception as e:
        print(f"\n[Warning] Could not save histogram: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEF_IN)
    ap.add_argument("--features", default=DEF_FEAT)
    ap.add_argument("--maxlen", type=int, default=DEF_MAXLEN)
    ap.add_argument("--thr", type=float, default=DEF_THR)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    main(args.csv, args.features, args.maxlen, args.thr, args.limit)

