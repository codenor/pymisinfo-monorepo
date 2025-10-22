#!/usr/bin/env python
import os, json, argparse, math, random
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, roc_auc_score

# ---- data loading (expects outputs from text_preprocessing.py) ----
DEF_FEAT = "./assets/features"

def load_splits(feature_dir: str):
    tr = np.load(os.path.join(feature_dir, "train_data.npz"))
    va = np.load(os.path.join(feature_dir, "val_data.npz"))
    te = np.load(os.path.join(feature_dir, "test_data.npz"))
    return (tr["X"], tr["y"]), (va["X"], va["y"]), (te["X"], te["y"])

def load_tokenizer(feature_dir: str):
    return joblib.load(os.path.join(feature_dir, "tokenizer.pkl"))

# ---- metrics helpers ----
def best_threshold_f1(y_true: np.ndarray, probs: np.ndarray):
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = (2*p*r) / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    # precision_recall_curve returns len(t) = len(p) - 1, hence i-1 guard
    thr = float(t[max(0, i-1)]) if len(t) > 0 else 0.5
    return thr, float(np.nanmax(f1)), float(p[i]), float(r[i])

# ---- model factory ----
def build_bilstm(vocab_size: int, maxlen: int,
                 embed_dim: int, lstm_units: int,
                 merge_mode: str = "concat",
                 dropout: float = 0.2, recurrent_dropout: float = 0.2):
    inp = layers.Input(shape=(maxlen,), dtype="int32")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout),
        merge_mode=merge_mode
    )(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)

# ---- training+eval for a single config ----
def train_and_eval(cfg, X_train, y_train, X_val, y_val, vocab_size, feature_dir, trial_id):
    # LR / optimizer
    opt = optimizers.Adam(learning_rate=cfg["lr"])

    model = build_bilstm(
        vocab_size=vocab_size,
        maxlen=int(X_train.shape[1]),
        embed_dim=cfg["embed_dim"],
        lstm_units=cfg["lstm_units"],
        merge_mode=cfg["merge_mode"],
        dropout=cfg["dropout"],
        recurrent_dropout=cfg["recurrent_dropout"],
    )
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=[metrics.BinaryAccuracy(name="accuracy"),
                           metrics.AUC(name="auc")])

    # class weights (optional)
    class_weight = None
    if cfg.get("class_weight", True):
        classes = np.unique(y_train)
        if len(classes) == 2:
            w = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight = {int(c): float(wi) for c, wi in zip(classes, w)}

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]

    # train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=cbs,
        class_weight=class_weight,
        verbose=0,
    )

    # val probabilities
    probs = model.predict(X_val, verbose=0).ravel()
    val_auc = float(roc_auc_score(y_val, probs))
    thr, f1, prec, rec = best_threshold_f1(y_val, probs)

    # save trial model if it’s the best later (caller decides)
    trial_path = os.path.join(feature_dir, f"bilstm_trial_{trial_id}.keras")
    model.save(trial_path)

    results = {
        "val_auc": val_auc,
        "val_best_f1": f1,
        "val_precision_at_best": prec,
        "val_recall_at_best": rec,
        "val_threshold": thr,
        "trial_model_path": trial_path,
    }
    return results

# ---- search space & sampler ----
DEFAULT_SPACE = {
    "embed_dim":        [64, 128, 192],
    "lstm_units":       [96, 128, 160],
    "merge_mode":       ["concat", "sum"],   # concat is wider; sum is smaller/faster
    "dropout":          [0.1, 0.2, 0.3],
    "recurrent_dropout":[0.0, 0.1, 0.2],
    "lr":               [1e-3, 5e-4, 2e-4],
    "batch_size":       [64, 96, 128],
    "epochs":           [8, 12, 16],
    "class_weight":     [True],              # toggle to [True, False] if you want to try both
}

def grid(space):
    # full cartesian grid (can explode)
    from itertools import product
    keys = list(space.keys())
    for values in product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))

def random_sample(space, max_trials, rng):
    keys = list(space.keys())
    for _ in range(max_trials):
        yield {k: rng.choice(space[k]) for k in keys}

# ---- main search ----
def main(feature_dir: str, max_trials: int, mode: str, seed: int):
    rng = np.random.default_rng(seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(feature_dir)
    tok = load_tokenizer(feature_dir)
    vocab_size = len(tok.word_index) + 1

    # sanity align maxlen from data
    maxlen = int(X_train.shape[1])

    # choose iterator
    space = DEFAULT_SPACE
    if mode == "grid":
        combo_iter = grid(space)
    else:
        combo_iter = random_sample(space, max_trials, rng)

    best = None
    best_idx = -1
    tried = 0

    for i, cfg in enumerate(combo_iter, 1):
        tried += 1
        # force epochs small if grid is huge
        if mode == "grid" and tried > max_trials > 0:
            break

        # train & eval a single trial
        res = train_and_eval(cfg, X_train, y_train, X_val, y_val, vocab_size, feature_dir, trial_id=i)

        score = (res["val_auc"], res["val_best_f1"])  # primary AUC, secondary F1
        if best is None or score > (best["val_auc"], best["val_best_f1"]):
            # delete prior best checkpoint to save space
            if best is not None and os.path.isfile(best["trial_model_path"]):
                try: os.remove(best["trial_model_path"])
                except: pass
            best = {**cfg, **res, "maxlen": maxlen, "vocab_size": vocab_size}
            best_idx = i

        print(f"[{i}] val_auc={res['val_auc']:.4f}  f1={res['val_best_f1']:.4f}  thr={res['val_threshold']:.3f}")

        if mode != "grid" and tried >= max_trials:
            break

    # finalize: rename best model, write settings
    assert best is not None, "No trials completed."
    best_path = os.path.join(feature_dir, "bilstm_best_tuned.keras")
    # move/overwrite
    try:
        if os.path.isfile(best_path): os.remove(best_path)
        os.replace(best["trial_model_path"], best_path)
    except Exception:
        # fallback copy
        tf.keras.models.load_model(best["trial_model_path"]).save(best_path)

    best["trial_model_path"] = best_path
    with open(os.path.join(feature_dir, "bilstm_tuned_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # evaluate on test with best threshold from val
    model = tf.keras.models.load_model(best_path)
    probs_test = model.predict(X_test, verbose=0).ravel()
    test_auc = float(roc_auc_score(y_test, probs_test))
    thr = float(best["val_threshold"])
    preds_test = (probs_test >= thr).astype(int)
    test_acc = float((preds_test == y_test).mean())

    summary = {
        "trials": tried,
        "selected_trial": best_idx,
        "val_auc": best["val_auc"],
        "val_best_f1": best["val_best_f1"],
        "val_threshold": thr,
        "test_auc_at_valthr": test_auc,
        "test_acc_at_valthr": test_acc,
        "model_path": best_path,
        "params_path": os.path.join(feature_dir, "bilstm_tuned_params.json"),
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=DEF_FEAT)
    ap.add_argument("--trials", type=int, default=12, help="number of random trials (ignored for full grid)")
    ap.add_argument("--mode", choices=["random","grid"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.features, max_trials=args.trials, mode=args.mode, seed=args.seed)
