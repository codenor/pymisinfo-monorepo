#!/usr/bin/env python
import os, argparse, numpy as np, joblib, json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences

DEF_FEAT = "./assets/features"
DEF_MAXLEN = 100

def load(feat):
    te = np.load(os.path.join(feat,"test_data.npz"))
    model = tf.keras.models.load_model(os.path.join(feat,"bilstm_best.keras"))
    tok   = joblib.load(os.path.join(feat,"tokenizer.pkl"))
    return (te["X"], te["y"]), model, tok

def pick_threshold(y_true, probs):
    # Sweep to maximize F1 (you can swap to precision/recall if preferred)
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = (2*prec*rec)/(prec+rec+1e-8)
    i = np.nanargmax(f1)
    return float(thr[max(0, i-1)]), {"best_f1": float(np.nanmax(f1)),
                                      "precision_at_best": float(prec[i]),
                                      "recall_at_best": float(rec[i])}

def main(feat, maxlen):
    (X_test, y_test), model, tok = load(feat)
    # sanity: align maxlen if needed
    if X_test.shape[1] != maxlen:
        maxlen = int(X_test.shape[1])

    # probs
    probs = model.predict(X_test, verbose=0).ravel()
    roc = roc_auc_score(y_test, probs)
    # threshold
    thr, thr_stats = pick_threshold(y_test, probs)
    preds = (probs >= thr).astype(int)

    # reports
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    out = {
        "test_auc": float(roc),
        "threshold": thr,
        "threshold_stats": thr_stats,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=DEF_FEAT)
    p.add_argument("--maxlen", type=int, default=DEF_MAXLEN)
    args = p.parse_args()
    main(args.features, args.maxlen)

