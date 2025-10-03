#!/usr/bin/env python

import pandas as pd
from pathlib import Path

# Ensures these dir are created
Path("assets/processed").mkdir(parents=True, exist_ok=True)
Path("assets/features").mkdir(parents=True, exist_ok=True)


def preprocess():
    # Raw data
    fake = pd.read_csv("assets/vicuni/Fake.csv")
    true = pd.read_csv("assets/vicuni/True.csv")

    # Add labels
    fake["label"] = 1
    true["label"] = 0

    # Merge and clean
    df = pd.concat([fake, true], ignore_index=True)
    df = df[["text", "label"]]
    df = df.dropna().drop_duplicates()

    # Save processed dataset to data dir
    df.to_csv("assets/processed/misinfo_dataset.csv", index=False)
    print("Saved to assets/processed/new-misinfo-dataset.csv")


if __name__ == "__main__":
    preprocess()
