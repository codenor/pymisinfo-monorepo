import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)


def evaluate_model(model_path, vec_path, data_path, model_name="Logistic Regression"):
    # Load model and vectoriser
    model = joblib.load(model_path)
    vectoriser = joblib.load(vec_path)

    # Load dataset
    df = pd.read_csv(data_path)
    X = df["claim"].astype(str)
    y = df["label"]

    # Transform
    X_tfidf = vectorizer.transform(X)

    # Predictions
    y_pred = model.predict(X_tfidf)
    y_proba = model.predict_proba(X_tfidf)[:, 1]

    # Metrics
    score = model.score(X_tfidf, y)
    print(f"\n- - - - - {model_name} Evaluation - - - - -")
    print(f"Accuracy: {score:.4f} ({score * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=3))

    # Plots
    os.makedirs("./assets/plots", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["True", "Misinformation"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"./assets/plots/{model_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="red", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./assets/plots/{model_name.lower().replace(' ', '_')}_roc.png")
    plt.close()

    # Class Distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.title("Label Distribution (0=True, 1=Misinformation)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("./assets/plots/label_distribution.png")
    plt.close()

    print(f"Plots saved in ./assets/plots/")
    return score


if __name__ == "__main__":
    evaluate_model(
        model_path="./assets/logreg_model.pkl",
        vec_path="./assets/features/vectoriser.pkl",
        data_path="./assets/processed/misinfo_dataset.csv",
        model_name="Logistic Regression",
    )
