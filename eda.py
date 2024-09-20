import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as  sns

df1 = pd.read_csv("datasets/australian_credit.csv")
df2 = pd.read_csv("datasets/GMSC/cs-training.csv")
df3 = pd.read_csv("datasets/german_credit.csv")
df4 = pd.read_csv("datasets/UCI_Credit_Card.csv")

datasets = [df1, df2, df3, df4]
pd.set_option('display.max_columns', None)
dataset_names = ["Australian Credit", "GMSC", "German Credit", "UCI Credit Card"]
label_dict = {0: "Non Default", 1: "Default"}

for i, df in enumerate(datasets):
    print(dataset_names[i], "\n")
    print(df.describe(include='all'))
    pd.notna(df).sum()

# split each dataset into X and y
Xs = []
ys = []

for dataset in datasets:
    Xs.append(dataset.iloc[:, :-1])
    ys.append(dataset.iloc[:, -1])

ys[2] = ys[2].replace({1:0, 2:1})

# Visualizing class imbalance
for y in ys:
    default_count = non_default_count = 0
    for i in range(len(y)):
        if y[i] == 0:
            non_default_count += 1
        else:
            default_count += 1
    print(default_count, non_default_count)

# Looking for categorical and binary variables
categorical_columns = [[],[],[],[]]
binary_columns = [[],[],[],[]]
for i, X in enumerate(Xs):
    for j in range(X.shape[1]):
        x = X.iloc[:, j].unique()
        if len(x) < 13 and len(x) > 2:
            categorical_columns[i].append(j)
        elif len(x) == 2:
            binary_columns[i].append(j)

categorical_columns
binary_columns


for i, X in enumerate(Xs):
    if i != 1:
        features = X.columns
        fig, axes = plt.subplots(nrows=round(len(categorical_columns[i])/2), ncols=2, figsize=(10, 10))
        for col, ax in zip(categorical_columns[i], axes.flat):
            for lab, color in zip(range(2), ('red', 'blue')):
                sns.countplot(x=X.loc[ys[i] == lab, features[col]],
                color=color,
                label=f"Class {label_dict[lab]}",
                alpha=0.6,
                ax=ax,
                )
        plt.suptitle(f"{dataset_names[i]} Categorical Variables")
        plt.show()

for i, X in enumerate(Xs):
    if i == 3:
        features = X.columns
        for lab, color in zip(range(2), ('red', 'blue')):
            sns.countplot(x=X.loc[ys[3] == lab, features[3]],
            color=color,
            label=f"Class {label_dict[lab]}",
            alpha=0.6
            )
        plt.suptitle(f"{dataset_names[i]} Categorical Variables")
        plt.show()
    if i in [0,2]:
        features = X.columns
        fig, axes = plt.subplots(nrows=round(len(binary_columns[i])/2), ncols=2, figsize=(10, 10))
        for col, ax in zip(binary_columns[i], axes.flat):
            for lab, color in zip(range(2), ('red', 'blue')):
                sns.countplot(x=X.loc[ys[i] == lab, features[col]],
                color=color,
                label=f"Class {label_dict[lab]}",
                alpha=0.6,
                ax=ax,
                )
        plt.suptitle(f"{dataset_names[i]} Categorical Variables")
        plt.show()

# Find numeric features
not_numeric_columns = [[],[],[],[]]
for i in range(4):
    not_numeric_columns[i] = categorical_columns[i] + binary_columns[i]
numeric_columns_index = [[],[],[],[]]
for i, X in enumerate(Xs):
    numeric_columns_index[i] = [x for x in range(X.shape[1]) if x not in not_numeric_columns[i]]


for i, X in enumerate(Xs):
    ncol = len(numeric_columns_index[i])
    features = X.iloc[:,numeric_columns_index[i]].columns
    fig, axes = plt.subplots(nrows=round(ncol/2), ncols=2, figsize=(10, 10))
    fig.subplots_adjust(hspace=2.2)
    for ax, cnt in zip(axes.ravel(), range(ncol)):
        for lab, color in zip(range(2), ('red', 'blue')):
            ax.hist(X.loc[ys[i] == lab, features[cnt]],
                    color=color,
                    label=f"Class {label_dict[lab]}",
                    alpha=0.6
                    )
        ax.set_xlabel(features[cnt])
        # hide axis ticks
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off",
                    labelleft="on")
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    red_patch = mpatches.Patch(color='red', label='Default', alpha=0.6)
    blue_patch = mpatches.Patch(color='blue', label='Non Default', alpha=0.6)

    fig.legend(handles=[red_patch, blue_patch], loc='center right', fancybox=True, fontsize=6)
    fig.suptitle("Numeric Features of " + dataset_names[i])
    plt.show()

# Handling missing values
for i, df in enumerate(datasets):
    if df.isna().sum().any():
        print(dataset_names[i])
        print(df.isna().sum()) # Only GMSC has missing values


df2[["MonthlyIncome", "NumberOfDependents"]].describe() # Using median to fill missing values seems appropriate

import numpy as np
# Correlation matrix
for df in datasets:
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True)
        plt.title(f"{dataset_names[i]} Correlation Matrix")
        plt.show()

df3.iloc[:, categorical_columns[2]] = pd.Categorical(df3.iloc[:,categorical_columns[2]]) # Checking for missing values in German Credit dataset
# Cast categorical columns to categorical type
pd.Categorical(df3.iloc[:, categorical_columns[2]])