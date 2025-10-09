#!/usr/bin/env python
# ── Quiet TensorFlow before it loads ───────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TF INFO & WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # stop oneDNN chatter

import re
import argparse
import numpy as np
import pandas as pd
import nltk
from ftfy import fix_text
from tensorflow.keras.preprocessing.text import Tokenizer

# --- cleaner matches model_2/text_preprocessing.py ---
NEG_KEEP = {"not", "no", "nor", "never", "without"}
RE_HTML = re.compile(r"<.*?>")
RE_PUNC = re.compile(r"[^\w\s'-]")  # keep apostrophes & hyphens

def load_stopwords():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    return {w for w in sw if w not in NEG_KEEP}

def clean_text(s: str, stop_words: set[str]) -> str:
    s = "" if not isinstance(s, str) else s
    s = fix_text(s)                 # fix mojibake/encoding glitches
    s = RE_HTML.sub("", s)
    s = RE_PUNC.sub("", s)          # DO NOT strip digits
    s = s.lower().strip()
    if not s:
        return s
    return " ".join(w for w in s.split() if w not in stop_words)

def suggest_maxlen(
    in_path: str = "./assets/processed/misinfo_dataset.csv",
    percentile: float = 95.0,
    clamp_min: int = 32,
    clamp_max: int = 512,
) -> dict:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    # robust CSV read (ignore extra cols like 'source')
    try:
        df = pd.read_csv(in_path, usecols=["claim", "label"], encoding="utf-8", encoding_errors="replace")
    except TypeError:
        df = pd.read_csv(in_path, usecols=["claim", "label"], encoding="utf-8")
    df = df.dropna().reset_index(drop=True)

    stop_words = load_stopwords()
    cleaned = df["claim"].astype(str).map(lambda x: clean_text(x, stop_words)).tolist()

    # Tokenize to get sequence lengths (no need to save tokenizer here)
    tok = Tokenizer(oov_token="<OOV>")
    tok.fit_on_texts(cleaned)
    seqs = tok.texts_to_sequences(cleaned)
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    if lengths.size == 0:
        raise ValueError("No non-empty texts after cleaning; cannot compute lengths.")

    stats = {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }

    raw_rec = int(np.percentile(lengths, percentile))
    recommended = int(np.clip(raw_rec, clamp_min, clamp_max))

    return {
        "recommended_maxlen": recommended,
        "percentile_used": percentile,
        "raw_percentile_value": raw_rec,
        "clamp_range": [clamp_min, clamp_max],
        "stats": stats,
    }

def _print_report(r: dict):
    s = r["stats"]
    print("Token length stats (after cleaning/tokenizing):")
    print(f"  count={s['count']}  min={s['min']}  mean={s['mean']:.2f}  median={s['median']:.0f}")
    print(f"  p90={s['p90']}  p95={s['p95']}  p99={s['p99']}  max={s['max']}")
    print(f"\nRecommended maxlen @ p{int(r['percentile_used'])}: {r['raw_percentile_value']}")
    print(f"Clamped to range {tuple(r['clamp_range'])}: **{r['recommended_maxlen']}**")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="./assets/processed/misinfo_dataset.csv")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--min", dest="clamp_min", type=int, default=32)
    ap.add_argument("--max", dest="clamp_max", type=int, default=512)
    args = ap.parse_args()

    report = suggest_maxlen(
        in_path=args.in_path,
        percentile=args.percentile,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
    )
    _print_report(report)

