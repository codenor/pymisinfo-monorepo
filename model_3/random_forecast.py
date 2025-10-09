#!/usr/bin/env python

import os
import sys

import rf_charts
import training

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "rf_model")
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")


def list_models():
    dir = os.listdir(MODELS_DIR)
    for d in dir:
        print(d)


def help():
    print("./logreg_supervised.py <command>")
    print("possible commands:")
    print("  train <model-name>          train a new model")
    print("  train-optimal-tree-count    train and plot the optimal tree count")
    print("  test <model-name>           test an existing model")
    print("  list                        list available models")
    print(
        "  chart <model-name> <name>   plot diagnostics (name = all|feature_importances|confusion_matrix|learning_curve)"
    )
    sys.exit(-1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help()
    cmd = sys.argv[1]
    match cmd:
        case "train":
            if len(sys.argv) < 3:
                help()
            training.train(sys.argv[2])
        case "train-optimal-tree-count":
            training.chart_tree_count_affect_rsme()
        case "test":
            if len(sys.argv) < 3:
                help()
            training.test(sys.argv[2])
        case "list":
            list_models()
        case "chart":
            if len(sys.argv) < 4:
                print("  Usage: ./logreg_supervised.py chart <model-name> <chart-name>")
                print(
                    "     <chart-name> can be one of: all, feature_importances, confusion_matrix, learning_curve"
                )
                sys.exit(-1)
            rf_charts.chart(sys.argv[2], sys.argv[3])
        case _:
            help()
