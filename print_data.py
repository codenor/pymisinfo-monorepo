import pandas as pd


def print_stuff():
    fake = pd.read_csv("assets/vicuni/Fake.csv")
    true = pd.read_csv("assets/vicuni/True.csv")

    processed = pd.read_csv("assets/processed/misinfo_dataset.csv")

    print(fake.head())
    print(true.head())
    print(processed.head())


if __name__ == "__main__":
    print_stuff()
