import os
import json
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

DATA_DIR = '/nlp/scr/kathli/eval/2hr'
OUTPUT_DIR = DATA_DIR

df = pd.read_csv(f'{DATA_DIR}/run_df.csv')

# Set custom tick labels for the x and y axes
x_labels = sorted(df["ShortLen"].unique())
y_labels = sorted(df["Ratio"].unique())

plt_titles = ["Avg Loss", "Avg Loss of last 255 tokens", "Variance of Loss"]
model_types = ["mixin", "2stage"]

# Create your data (3x3 array)
data = np.zeros((len(plt_titles) * len(model_types), len(y_labels), len(x_labels)))

# loop through the rows of the df
for i, row in df.iterrows():
    name = row["Name"]
    short_len = row["ShortLen"]
    ratio = row["Ratio"]
    row_type = row["Type"]
    type_idx = model_types.index(row_type)
    with open(f"{DATA_DIR}/eval-loss-openwebtext_{name}.json", 'rb') as f:
        data_dict = json.load(f)
        L = data_dict["mean"]
        V = data_dict["variance"]
        L_last = data_dict["cutoff_mean"]

        print(f"{name} - {short_len} - {ratio} - {row_type} - {L} - {V} - {L_last}")

    offset = type_idx * len(plt_titles)
    data[offset][y_labels.index(ratio)][x_labels.index(short_len)] = L
    data[offset + 1][y_labels.index(ratio)][x_labels.index(short_len)] = L_last
    data[offset + 2][y_labels.index(ratio)][x_labels.index(short_len)] = V

plt.figure(figsize=(12, 8))

# Create a figure and a set of subplots
for i in range(len(model_types)):
    for j in range(len(plt_titles)):
        idx = i * len(plt_titles) + j
        plt.subplot(2, 3, idx+1)
        #print(data[idx,:,:])
        df = pd.DataFrame(data[idx,:,:], columns=x_labels, index=y_labels)
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
        plt.title(f"{plt_titles[j]} - {model_types[i]}")
        plt.xlabel("ShortLen")
        plt.ylabel("Ratio")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/eval-loss-openwebtext.png", dpi=300)

plt.show()

