#!/usr/bin/env python
# ── Quiet TensorFlow before it loads ───────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import nltk
import joblib
import numpy as np
import pandas as pd
import logging

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from ftfy import fix_text
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Hardcoded sequence length (from p95)
MAXLEN = 273

# ---------- IO helpers ----------
def _read_csv_claim_label(path: str) -> pd.DataFrame:
    """Read only claim/label with robust UTF-8 handling and ftfy fix."""
    try:
        df = pd.read_csv(path, usecols=["claim", "label"], encoding="utf-8", encoding_errors="replace")
    except TypeError:
        df = pd.read_csv(path, usecols=["claim", "label"], encoding="utf-8")
    df = df.dropna().reset_index(drop=True)
    df["claim"] = df["claim"].astype(str).map(fix_text)
    return df

# ---------- summary helpers ----------
def _class_counts_only(y: np.ndarray) -> dict:
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}

def preprocess(
    in_path: str = "./assets/processed/misinfo_dataset.csv",
    out_dir: str = "./assets/features",
    maxlen: int = MAXLEN,
    random_state: int = 42,
):
    # stopwords (keep negations)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    for w in {"not", "no", "nor", "never", "without"}:
        stop_words.discard(w)

    # load & validate
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")
    df_raw = _read_csv_claim_label(in_path)
    n0 = len(df_raw)
    if df_raw.empty:
        raise ValueError("No valid rows after dropping NaNs in 'claim'/'label'.")

    # clean (keep numbers; allow apostrophes & hyphens)
    re_html = re.compile(r"<.*?>")
    re_punc = re.compile(r"[^\w\s'-]")  # keep ' and -

    def clean_text(s: str) -> str:
        s = "" if not isinstance(s, str) else s
        s = re_html.sub("", s)
        s = re_punc.sub("", s)          # keep digits
        s = s.lower().strip()
        if not s:
            return s
        return " ".join(w for w in s.split() if w not in stop_words)

    df = df_raw.copy()
    df["cleaned"] = df["claim"].astype(str).map(clean_text)

    # drop rows that became empty after cleaning
    mask_nonempty = df["cleaned"].str.strip().astype(bool)
    emptied = int((~mask_nonempty).sum())
    df = df[mask_nonempty].reset_index(drop=True)

    # --- load labels and flip: CSV defines 1=factual, 0=misinfo ---
    y = pd.to_numeric(df["label"], errors="raise").to_numpy()
    y = 1 - y  # now 1 = misinformation, 0 = factual ✅

    # split FIRST (avoid vocab leakage)
    texts = df["cleaned"].tolist()
    X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
        texts, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val_txt, X_test_txt, y_val, y_test = train_test_split(
        X_temp_txt, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # tokenizer on TRAIN ONLY
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    # to sequences
    X_train = tokenizer.texts_to_sequences(X_train_txt)
    X_val   = tokenizer.texts_to_sequences(X_val_txt)
    X_test  = tokenizer.texts_to_sequences(X_test_txt)

    # pad
    X_train = pad_sequences(X_train, padding="post", truncating="post", maxlen=maxlen)
    X_val   = pad_sequences(X_val,   padding="post", truncating="post", maxlen=maxlen)
    X_test  = pad_sequences(X_test,  padding="post", truncating="post", maxlen=maxlen)

    # save artifacts
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(out_dir, "val_data.npz"),   X=X_val,   y=y_val)
    np.savez_compressed(os.path.join(out_dir, "test_data.npz"),  X=X_test,  y=y_test)
    joblib.dump(tokenizer, os.path.join(out_dir, "tokenizer.pkl"))

    # compact summary (for preview / report)
    dropped_total = (n0 - len(df_raw)) + emptied  # dropna + emptied-after-clean
    summary = {
        "csv": os.path.abspath(in_path),
        "maxlen": int(maxlen),
        "vocab_size": int(len(tokenizer.word_index) + 1),
        "rows_total": int(n0),
        "rows_used": int(len(df)),
        "rows_dropped": int(dropped_total),
        "splits": {
            "train": {"n": int(len(y_train)), "classes": _class_counts_only(y_train)},
            "val":   {"n": int(len(y_val)),   "classes": _class_counts_only(y_val)},
            "test":  {"n": int(len(y_test)),  "classes": _class_counts_only(y_test)},
        },
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer, summary

# --------- preview (concise, non-technical) ---------
def print_preview(summary, tokenizer, maxlen, df_path):
    """Plain, non-technical preview without any example transformation."""
    total   = summary["rows_total"]
    used    = summary["rows_used"]
    dropped = summary["rows_dropped"]
    tr = summary["splits"]["train"]["n"]
    va = summary["splits"]["val"]["n"]
    te = summary["splits"]["test"]["n"]

    print("\n=== Data ready for the model ===")
    print(f"- Source file: {summary['csv']}")
    print(f"- Total rows in file: {total}")
    print(f"- Rows used: {used}   (dropped {dropped} empty or unusable rows)")
    print(f"- Split into:  Train={tr}  Validate={va}  Test={te}")
    print(f"- Each sentence becomes exactly {maxlen} numbers.")
    print(f"- Word list size (vocabulary): {summary['vocab_size']}\n")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="./assets/processed/misinfo_dataset.csv")
    p.add_argument("--out", dest="out_dir", default="./assets/features")
    p.add_argument("--maxlen", type=int, default=MAXLEN)
    p.add_argument("--preview", action="store_true", help="print a short summary (no examples)")
    args = p.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), tok, summary = preprocess(
        in_path=args.in_path, out_dir=args.out_dir, maxlen=args.maxlen
    )

    # Save a JSON report alongside artifacts
    report_path = os.path.join(args.out_dir, "preprocess_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved report → {report_path}")

    if args.preview:
        print_preview(summary, tok, args.maxlen, args.in_path)

