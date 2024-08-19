import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

df1 = pd.read_csv("datasets/australian_credit.csv")
df2 = pd.read_csv("datasets/GMSC/cs-training.csv")
df3 = pd.read_csv("datasets/german_credit.csv")
df4 = pd.read_csv("datasets/UCI_Credit_Card.csv")

datasets = [df1, df2, df3, df4]

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
for X in Xs:
    for i in range(X.shape[1]):
        x = X.iloc[:, i].unique()
        if len(x) < 15:
            print(x, i)
    print("next")

# Encode categorical columns which are not binary
ct = ColumnTransformer()