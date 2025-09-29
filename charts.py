#!/usr/bin/env python

import pandas
import matplotlib.pyplot as pyplot

df = pandas.read_csv("./assets/preprocessed-training.csv")

bad_chars = [ "donald trump", "covid-19", "covid 19", "coronavirus", "trump" ]

# content,username,upload_date,category,misinformation_type,datasource,hedge_words_count,symobls,all_caps
misinfo_type = df.iloc[:, 4]
hedge_char_count = df.iloc[:, 6]
symbol_count = df.iloc[:, 7]
all_caps_count = df.iloc[:, 8]
# fk_scores = df.iloc[:, 9].astype(float)

grouped_hedge_count = {}
grouped_symbol_count = {}
grouped_all_caps_count = {}
for misinfo_type, hedge_c, symbol_c, caps_c in zip(misinfo_type, hedge_char_count, symbol_count, all_caps_count):
    exists = False
    grouped_hedge_count.setdefault(misinfo_type, []).append(hedge_c)
    grouped_symbol_count.setdefault(misinfo_type, []).append(symbol_c)
    grouped_all_caps_count.setdefault(misinfo_type, []).append(caps_c)
    # grouped_fk_scores.setdefault(misinfo_type, []).append(fk_scores_raw)

symbol_labels = list(grouped_symbol_count.keys())
hedge_labels = list(grouped_hedge_count.keys())
all_caps_labels = list(grouped_all_caps_count.keys())
# fk_scores_labels = list(grouped_fk_scores.keys())

hedge_chart_data = list(grouped_hedge_count.values())
symbol_chart_data = list(grouped_symbol_count.values())
all_caps_data = list(grouped_all_caps_count.values())
# fk_scores_data = list(grouped_fk_scores.values())

fig, (symbol_chart, hedge_chart, caps_chart, fk_chart) = pyplot.subplots(1, 4, figsize=(8, 6))

symbol_chart.boxplot(symbol_chart_data, labels=symbol_labels)
symbol_chart.set_xlabel("Is Misinformation")
symbol_chart.set_ylabel("Amount Of Symbols")
symbol_chart.set_title("Correlation between symbols and type of misinformation")
symbol_chart.tick_params(axis='x', rotation=45)

hedge_chart.boxplot(hedge_chart_data, labels=hedge_labels)
hedge_chart.set_xlabel("Is Misinformation")
hedge_chart.set_ylabel("Amount Of Hedge Characters")
hedge_chart.set_title("Correlation between hedge characters and type of misinformation")
hedge_chart.tick_params(axis='x', rotation=45)

caps_chart.boxplot(all_caps_data, labels=all_caps_labels)
caps_chart.set_xlabel("Is Misinformation")
caps_chart.set_ylabel("Amount of words that contain all-caps, and have more than 1 character")
caps_chart.set_title("Correlation between all-caps words and type of misinformation")
caps_chart.tick_params(axis='x', rotation=45)

# caps_chart.boxplot(fk_scores_data, labels=fk_scores_labels)
# caps_chart.set_xlabel("Is Misinformation")
# caps_chart.set_ylabel("Flesch Kincaid Score")
# caps_chart.set_title("Correlation between Flesch-Kincaid score and type of misinformation")
# caps_chart.tick_params(axis='x', rotation=45)

pyplot.tight_layout()

pyplot.show()
