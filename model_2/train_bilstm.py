#!/usr/bin/env python
import os
# Silence TF/XLA/CUDA banners
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # 0=all,1=INFO off,2=+WARN off,3=+ERROR off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # stop oneDNN chatter
os.environ["CUDA_VISIBLE_DEVICES"] = ""           # force CPU (prevents cuInit messages)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import argparse
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics
from sklearn.utils.class_weight import compute_class_weight

# Defaults assume your artifacts from text_preprocessing.py
DEFAULT_FEATURE_DIR = "./assets/features"
DEFAULT_MAXLEN = 273  # must match your preprocessing maxlen

def load_splits(feature_dir: str):
    tr = np.load(os.path.join(feature_dir, "train_data.npz"))
    va = np.load(os.path.join(feature_dir, "val_data.npz"))
    te = np.load(os.path.join(feature_dir, "test_data.npz"))
    return (tr["X"], tr["y"]), (va["X"], va["y"]), (te["X"], te["y"])

def load_tokenizer(feature_dir: str):
    return joblib.load(os.path.join(feature_dir, "tokenizer.pkl"))

def build_bilstm(vocab_size: int, maxlen: int, embed_dim: int = 192, lstm_units: int = 160):
    model = models.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=maxlen,
            mask_zero=True,        # ignore padding tokens (0)
        ),
        layers.Bidirectional(layers.LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")  # binary classification
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[metrics.BinaryAccuracy(name="accuracy"), metrics.AUC(name="auc")]
    )
    return model

def compute_weights_if_binary(y_train: np.ndarray):
    classes = np.unique(y_train)
    if len(classes) == 2:
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return {int(c): float(wi) for c, wi in zip(classes, w)}
    return None

def main(
    feature_dir: str = DEFAULT_FEATURE_DIR,
    maxlen: int = DEFAULT_MAXLEN,
    epochs: int = 20,
    batch_size: int = 96,
    use_class_weight: bool = True,
    out_model_path: str | None = None,
):
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(feature_dir)
    tok = load_tokenizer(feature_dir)
    vocab_size = len(tok.word_index) + 1  # +1 for padding index 0

    # Sanity check maxlen
    if X_train.shape[1] != maxlen:
        print(f"[warn] maxlen mismatch: data={X_train.shape[1]} vs arg={maxlen}. Using data shape.")
        maxlen = int(X_train.shape[1])

    # Build model
    model = build_bilstm(vocab_size=vocab_size, maxlen=maxlen)

    # Callbacks
    ckpt_path = out_model_path or os.path.join(feature_dir, "bilstm_best.keras")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # Class weights (helps if imbalanced)
    class_weight = compute_weights_if_binary(y_train) if use_class_weight else None

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print({"test_loss": float(test_loss), "test_acc": float(test_acc), "test_auc": float(test_auc)})

    # Save final model (in addition to best checkpoint)
    final_path = os.path.join(feature_dir, "bilstm_final.keras")
    model.save(final_path)

    # Save a compact training history json
    hist_path = os.path.join(feature_dir, "bilstm_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    print(f"Saved best model → {ckpt_path}")
    print(f"Saved final model → {final_path}")
    print(f"Saved history → {hist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=DEFAULT_FEATURE_DIR, help="Directory with train/val/test .npz and tokenizer.pkl")
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="Must match preprocessing")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--no-class-weight", action="store_true", help="Disable class weighting")
    parser.add_argument("--out", dest="out_model_path", default=None, help="Path for best checkpoint (.keras)")
    args = parser.parse_args()

    main(
        feature_dir=args.features,
        maxlen=args.maxlen,
        epochs=args.epochs,
        batch_size=args.batch,
        use_class_weight=not args.no_class_weight,
        out_model_path=args.out_model_path,
    )

