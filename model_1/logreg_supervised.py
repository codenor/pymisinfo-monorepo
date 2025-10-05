import os
import argparse
import joblib
from typing import Optional
from utils import load_features, train_test_split_data, evaluate_supervised
from sklearn.linear_model import LogisticRegression


def run_logreg(custom_text: Optional[str] = None):
    # Load TF-IDF features + labels
    X, y = load_features()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train logistic regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    if custom_text:
        # Build absolute path to vectoriser.pkl
        vec_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "features", "vectoriser.pkl"
        )
        vec_path = os.path.abspath(vec_path)

        # Load vectorizer
        vectorizer = joblib.load(vec_path)

        # Transform and predict custom input
        X_custom = vectorizer.transform([custom_text])
        prediction = model.predict(X_custom)[0]

        print("\n- - - - - Custom Prediction - - - - -")
        print(f"Input: {custom_text}")
        print(f"Predicted label: {'FALSE' if prediction==0 else 'TRUE'}")
        print("- - - - - - - - - - - - - - - - - - -\n")
    else:
        # Evaluate normally
        print("- - - - - Logistic Regression - - - - - -")
        evaluate_supervised(model, X_test, y_test)
        print("- - - - - - - - - - - - - - - - - - - - - -")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Text Classifier")
    parser.add_argument("--text", type=str, help="Custom text")
    args = parser.parse_args()

    run_logreg(custom_text=args.text)
