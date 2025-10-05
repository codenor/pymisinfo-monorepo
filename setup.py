from preprocess import *
from vectorise import *
from print_data import *


def setup():
    print("Preprocessing . . .")
    preprocess()
    print("DONE!\n")

    print("Vectorising . . .")
    vectorise()
    print_stuff()
    print("DONE!")


if __name__ == "__main__":
    setup()
