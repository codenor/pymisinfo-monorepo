#!/usr/bin/env python

import argparse
import csv
import io
from datetime import date, datetime, time
import os
from typing import Any

import pandas
from numpy import uint
from pandas.io.common import file_exists
from rich.progress import Progress
from typing_extensions import Writer

DATE_FORMATS = [
    "%B %d, %Y",  # "January 2, 2006"
    "%d-%b-%y",  # "02-Jan-06"
    "%b %d, %Y",  # "Jan 02, 2006"
]

SYMBOLS = ["!", "...", "?"]


class ProgramArgs:
    def __init__(
        self,
        input_true_file: str = "./assets/vicuni/True.csv",
        input_fake_file: str = "./assets/vicuni/Fake.csv",
        output_file: str = "./assets/preprocess.csv",
        parse_csv_in_memory: bool = False,
        hedge_char_file: str = "./assets/hedge-words.txt",
        auto_accept: bool = False,
    ) -> None:
        self.input_true_file = input_true_file
        self.input_fake_file = input_fake_file
        self.output_file = output_file
        self.parse_csv_in_memory = parse_csv_in_memory
        self.hedge_char_file = hedge_char_file
        self.auto_accept = auto_accept


class CsvRecord:
    def __init__(
        self,
        content: str,
        username: str,
        upload_date: date,
        category: str,
        misinfo_type: str,
        datasource: str,
        hedge_chars_count: uint,
        symbols_count: uint,
        all_caps_count: uint,
    ):
        self.content = content
        self.username = username
        self.upload_date = upload_date
        self.category = category
        self.misinfo_type = misinfo_type
        self.datasource = datasource
        self.hedge_chars_count = hedge_chars_count
        self.symbols_count = symbols_count
        self.all_caps_count = all_caps_count

    def to_csv_list(self) -> list[str]:
        return [
            self.content,
            self.username,
            self.upload_date.strftime("%d/%m/%Y"),
            self.category,
            self.misinfo_type,
            self.datasource,
            str(self.hedge_chars_count),
            str(self.symbols_count),
            str(self.all_caps_count),
        ]


def try_strptime(date_str, formats: list[str]) -> datetime | None:
    for format in formats:
        try:
            time = datetime.strptime(date_str, format)
        except ValueError:
            continue
        return time
    return None


def get_hedge_words(file: str = "./hedge-words.txt") -> list[str]:
    with open(file, "r") as f:
        text = f.readline()
        words = text.split(',')
        idx = 0
        for w in words:
            words[idx] = w.strip().lower()
            idx += 1
        return words


def count_of_many_substrings(string: str, substrings: list[str]) -> uint:
    amount = uint(0)
    for s in substrings:
        amount += string.count(s)
    return amount


def all_caps_count(string: str) -> uint:
    amount = uint(0)
    words = string.split(" ")
    for word in words:
        is_all_caps = True
        for c in word:
            if c.islower():
                is_all_caps = False
                break
        if is_all_caps:
            amount += 1
    return amount


def process_record(
    record: tuple[int, Any, Any, Any, Any],
    misinformation_type: str,
    hedge_word_substrings: list[str],
    symbol_substrings: list[str],
) -> CsvRecord:
    title = str(record[1]).strip()
    title_lower = title.lower()
    # text = str(record[2]).strip()
    subject = str(record[3]).strip()
    date = try_strptime(str(record[4]).strip(), DATE_FORMATS)

    if date == None:
        raise ValueError(f"date {date} was an invalid format")

    hedge_char_c = count_of_many_substrings(title_lower, hedge_word_substrings)
    symbols_c = count_of_many_substrings(title_lower, symbol_substrings)
    all_caps_c = all_caps_count(title)

    return CsvRecord(
        title,
        "",
        date,
        subject,
        misinformation_type,
        "Victoria University",
        hedge_char_c,
        symbols_c,
        all_caps_c,
    )


def get_args() -> ProgramArgs:
    """Parses the command line arguments. Will call sys.exit() if the -h (help) flag is passed"""
    parser = argparse.ArgumentParser(
        prog="Misinformation Dataset Preprocessor (Victoria University Dataset)",
        description="Preprocesses (cleans) everything from the victoria university dataset into an output csv file",
    )

    parser.add_argument(
        "-t",
        "--input-true",
        dest="input_true",
        help="The True.csv file from the Victoria University misinformation dataset. Default='./assets/vicuni/True.csv'",
        default="./assets/vicuni/True.csv",
    )
    parser.add_argument(
        "-f",
        "--input-fake",
        dest="input_fake",
        help="The Fake.csv file from the Victoria University misinformation dataset. Default='./assets/vicuni/Fake.csv'",
        default="./assets/vicuni/Fake.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="The file to output the processed information. Default='./assets/preprocessed.csv'",
        default="./assets/preprocessed.csv",
    )
    parser.add_argument(
        "-m",
        "--in-memory",
        dest="in_memory",
        help="Whether the CSV processing should be done in-memory. You will need a lot of RAM on your system for this to work, but will have performance improvements. Default=False",
        default=False,
    )
    parser.add_argument(
        "-y",
        "--auto-accept",
        dest="auto_accept",
        help="Whether the program should automatically accept inputs (such as overwriting files). Default=False",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--hedge-word-file",
        dest="hedge_word_file",
        help="Path to the file containing comma-delimited hedge words. Default='./assets/hedge-words.txt'",
        default="./assets/hedge-words.txt",
    )

    arguments = parser.parse_args()

    return ProgramArgs(
        str(arguments.input_true),
        str(arguments.input_fake),
        str(arguments.output),
        bool(arguments.in_memory),
        str(arguments.hedge_word_file),
        bool(arguments.auto_accept),
    )


def process_file(
    hedge_words: list[str],
    input_file_path: str,
    misinfo_classification: str,
    output_file_writer,
    parse_in_memory: bool,
):
    output_file_csv = csv.writer(output_file_writer)
    csv_data = pandas.read_csv(
        input_file_path,
        header=1,
        engine="c",
        on_bad_lines="warn",
        dtype={
            "title": str,
            "text": str,
            "subject": str,
            "date": str,
        },
        memory_map=parse_in_memory,
    )

    with Progress() as p:
        t = p.add_task("generating documents to insert", total=len(csv_data))

        idx = 0
        for current in csv_data.itertuples():
            idx += 1
            try:
                record = process_record(
                    current, misinfo_classification, hedge_words, SYMBOLS
                )
            except ValueError as v:
                print(f"invalid record at line {idx}: {v}")
                continue

            output_file_csv.writerow(record.to_csv_list())

            p.advance(t)


def check_delete_file(path: str, auto_accept: bool=False):
    if file_exists(path):
        if auto_accept:
            os.remove(path)
        else:
            i = input(f"overwriting '{path}', Continue? [y/N] ")
            i = i.strip().lower()
            if i == 'y':
                os.remove(path)
            else:
                print("stopping")
                exit()



def main():
    args = get_args()
    hedge_words = get_hedge_words(args.hedge_char_file)

    check_delete_file(args.output_file, args.auto_accept)

    with open(args.output_file, "w") as output_file:
        output_file_csv = csv.writer(output_file)
        output_file_csv.writerow([
            "content",
            "username",
            "upload_date",
            "category",
            "misinformation_type",
            "datasource",
            "hedge_words_count",
            "symobls",
            "all_caps",
        ])

        process_file(
            hedge_words,
            args.input_true_file,
            "true",
            output_file,
            args.parse_csv_in_memory,
        )
        process_file(
            hedge_words,
            args.input_fake_file,
            "fake",
            output_file,
            args.parse_csv_in_memory,
        )


if __name__ == "__main__":
    main()
