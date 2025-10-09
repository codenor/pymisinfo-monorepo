# pymisinfo

## Training setup

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

### Confusion Matrix - supervised

- [Compute Classification Report and Confusion Matrix]("https://www.geeksforgeeks.org/machine-learning/compute-classification-report-and-confusion-matrix-in-python/")

### Clustering Scores - unsupervised

- [Rand-Index in Machine Learning](https://www.geeksforgeeks.org/machine-learning/rand-index-in-machine-learning/) (ARI)

- [Silhouette](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)
