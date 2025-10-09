#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values from your report
cm = np.array([[17781, 13434],
               [2812, 26070]])

labels = ["Factual (0)", "Misinformative (1)"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt=",",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 12, "weight": "bold"}
)
plt.title("Confusion Matrix (Threshold = 0.38)", fontsize=13, weight="bold", pad=10)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig("./assets/features/confusion_matrix_heatmap.png", dpi=200)
plt.close()
print("Saved → ./assets/features/confusion_matrix_heatmap.png")
