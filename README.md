# pymisinfo

## Training setup

### Requirements

- Python
- Pip

Install required libraries:
```
pip install -r requirements.txt
```

### Model 1 - Logistic Regression

Run TF-IDF vectoriser:

```
python vectoriser.py
```

Run logreg module to train model:
```
python -m model_1.logreg_supervised
```

---

- Used for data preparation and feature extraction for the detection of misinformation in text.
- Tools include; data preprocessing, vectorisation etc.
- vectorisation is done via TF-IDF for both supervised and unsupervised models.

`preprocess.py`

- Ensures dir are created.
- If datasets are split into different datasets it.
- Cleans text.
- Outputs in `text, label` form.

`print-data.py`

- Prints first rows of each dataset

`vectorise.py`

- Convert cleaned dataset (after `preprocess.py`) into TF-IDF vectors.
  - `tfidf-features.npz` - Matrix of text features.
  - `labels.csv` - label column (0, 1).
  - `vectoriser.pkl` - fitted TF-IDF model for reuse.

## Utils

Methos that can be used within your models.

```python
# Loads TF-IDF features and models
load_features()

# Splits dataset into training and test sets
train_test_split_data()

# Prints confusion matrix + report
evaluate_supervised()

# Prints clustering scores (ARI, silhouette)
evaluate_unsupervised()

```
## Model 2 – Bidirectional LSTM (Deep Learning)

A context-aware neural model that captures sequential language dependencies to better distinguish factual from misinformative text.

---

### Dataset Format

CSV input required at:

```
./assets/processed/misinfo_dataset.csv
```
---

### Directory Structure

```
model_2/
│
├── text_preprocessing.py   # Cleans, tokenises, splits train/val/test
├── suggest_maxlen.py       # Suggests optimal sequence length
├── train_bilstm.py         # Trains and saves BiLSTM
├── tune_bilstm.py          # Hyperparameter tuning
├── evaluate.py             # Evaluates test accuracy, F1, AUC
└── misclassified.py        # Shows false positives/negatives

assets/
├── processed/              # misinfo_dataset.csv
└── features/               # Models, tokeniser, feature splits
```

---

### Setup & Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy pandas joblib tensorflow ftfy nltk scikit-learn matplotlib
```

---

## Workflow for Model 2

### **(Optional) Suggest Sequence Length**

```bash
python -m model_2.suggest_maxlen --in ./assets/processed/misinfo_dataset.csv
```

Output example:

```
Token length stats (after cleaning/tokenizing):
  count=59318  min=3  mean=87.42  median=71
  p90=231  p95=273  p99=352  max=790
Recommended maxlen @ p95: 273
```

---

### **Preprocess Dataset**

```bash
python -m model_2.text_preprocessing --preview
```

Generates:

```
assets/features/
 ├── train_data.npz
 ├── val_data.npz
 ├── test_data.npz
 ├── tokenizer.pkl
 └── preprocess_report.json
```

---

### **Train BiLSTM**

```bash
python -m model_2.train_bilstm --epochs 12 --batch 64
```

Outputs:

```
assets/features/
 ├── bilstm_best.keras
 ├── bilstm_final.keras
 └── bilstm_history.json
```

---

### **Evaluate Model**

```bash
python -m model_2.evaluate
```

Example JSON:

```json
{
  "test_auc": 0.8556,
  "threshold": 0.33,
  "threshold_stats": {
    "best_f1": 0.7740,
    "precision_at_best": 0.7230,
    "recall_at_best": 0.8327
  },
  "confusion_matrix": [[18145, 9959], [5221, 25993]]
}
```

---

### **Inspect Misclassifications**

```bash
python -m model_2.misclassified --thr 0.38 --limit 10
```

Shows false negatives and false positives for manual inspection.

---

### **Tune Hyperparameters**

#### Random Search (recommended)

```bash
python -m model_2.tune_bilstm --trials 12 --mode random --seed 42
```

#### Grid Search (full)

```bash
python -m model_2.tune_bilstm --mode grid --trials 20
```

Produces:

```
assets/features/
 ├── bilstm_best_tuned.keras
 ├── bilstm_tuned_params.json
```

---

## Command Recap

```bash
# 1. (Optional)
python -m model_2.suggest_maxlen

# 2. Preprocess
python -m model_2.text_preprocessing --preview

# 3. Train
python -m model_2.train_bilstm --epochs 12 --batch 64

# 4. Evaluate
python -m model_2.evaluate

# 5. Inspect errors
python -m model_2.misclassified --thr 0.38 --limit 10

# 6. Tune hyperparameters
python -m model_2.tune_bilstm --trials 12 --mode random --seed 42
```

---

## Output Summary

| File                       | Description               |
| -------------------------- | ------------------------- |
| `bilstm_best.keras`        | Best checkpoint model     |
| `bilstm_tuned_params.json` | Optimal hyperparameters   |
| `preprocess_report.json`   | Preprocessing statistics  |
| `bilstm_history.json`      | Accuracy and loss history |
| `bilstm_best_tuned.keras`  | Final tuned BiLSTM model  |

---
### Confusion Matrix - supervised

- [Compute Classification Report and Confusion Matrix]("https://www.geeksforgeeks.org/machine-learning/compute-classification-report-and-confusion-matrix-in-python/")

### Clustering Scores - unsupervised

- [Rand-Index in Machine Learning](https://www.geeksforgeeks.org/machine-learning/rand-index-in-machine-learning/) (ARI)

- [Silhouette](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)
