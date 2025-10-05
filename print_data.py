import pandas as pd


def print_stuff():
    processed = pd.read_csv("assets/raw/claims.csv")
    print(processed.head())


if __name__ == "__main__":
    print_stuff()
