# Importing libraries
# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Prepocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# Performance evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix


# Load dataset
dataset = pd.read_csv("datasets/australian_credit.csv")
X = dataset.iloc[:, :-1]
y = np.ravel(dataset.iloc[:, -1])

# Visualizing class imbalance
default_count = non_default_count = 0
for i in range(len(y)):
    if y[i] == 0:
        non_default_count += 1
    else:
        default_count += 1

plt.Figure(figsize=(8, 8))
sns.set_style('darkgrid')
sns.barplot(x=['Non Default', 'Default'], y=[default_count, non_default_count],
            edgecolor='black', linewidth=1.5, hue=['Non Default', 'Default'])
plt.title('Defaults vs Non-Defaults')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Looking for categorical and binary variables
for i in range(14):
    x = pd.DataFrame(X).iloc[:, i].unique()
    if len(x) < 15:
        print(x, i)

# Encode categorical columns which are not binary
ct = ColumnTransformer(
    [('standardscaler', StandardScaler(), [1]),
     ("robustscaler", RobustScaler(), [2, 6, 9, 12, 13]),
     ("one_hot_encoder", OneHotEncoder(categories="auto"), [3, 4, 5, 11])],
    remainder='passthrough'
)
X = pd.DataFrame(ct.fit_transform(X))
X.columns = [f"A{i}" for i in range(1, 39)]
# Remove outliers
X_main = X.query('A1 < 3.1 & A3 < 10 & A4 < 10 & A5 < 7.5 & A6 < 200')

# Comparing boxplots of the data with and without outliers
fig, ax = plt.subplots(nrows=3, ncols=2)
for fig, cnt in zip(ax.ravel(), range(6)):
    fig.boxplot(X.iloc[:, cnt], vert=False)

fig, ax = plt.subplots(nrows=3, ncols=2)
for fig, cnt in zip(ax.ravel(), range(6)):
    fig.boxplot(X_main.iloc[:, cnt], vert=False)

# Histogram of the data
label_dict = {0: "Non Default", 1: "Default"}
feature_dict = {i: feature for i, feature in zip(range(24), dataset.columns)}

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10, 10))
fig.subplots_adjust(hspace=2.2)
for ax, cnt in zip(axes.ravel(), range(14)):
    min_b = np.min(dataset.iloc[:, cnt])
    max_b = np.max(dataset.iloc[:, cnt])
    bins = np.linspace(min_b, max_b, 25)

    for lab, col in zip(range(2), ('red', 'blue')):
        ax.hist(X.loc[y == lab, feature_dict[cnt]],
                color=col,
                label=f"Class {label_dict[lab]}",
                bins=bins,
                alpha=0.6
                )
    ylims = ax.get_ylim()
    ax.set_ylim([0, max(ylims) + 2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title("")
    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off",
                   labelleft="on")
    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

red_patch = mpatches.Patch(color='red', label='Rejected', alpha=0.6)
blue_patch = mpatches.Patch(color='blue', label='Approved', alpha=0.6)

fig.legend(handles=[red_patch, blue_patch], loc='center right',
           bbox_to_anchor=(1.1, 0.5), fancybox=True, fontsize=6)
fig.suptitle("Histogram of Features")
plt.show()

# Splitting dataset into train & test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_main, y[X_main.index], test_size=0.2, random_state=0
)

# Check distribution of the data without outliers
outliers = dataset.loc[X.index.difference(X_main.index), :].iloc[:, [
    1, 2, 6, 9, 12, 13]]
dataset.loc[X_main.index, :].describe(include="all")

print(f"Percentage of data remaining: {len(X_main) / len(X) * 100}%")  # 98.26%

fits = [LogisticRegression(penalty=None, solver="newton-cg"), 
        LinearDiscriminantAnalysis(), 
        KNeighborsClassifier(n_neighbors=7), 
        MLPClassifier(max_iter=1000), 
        RandomForestClassifier(),
        DecisionTreeClassifier(), 
        LinearSVC(dual="auto", max_iter=1000)]

accuracy_scores_test = []
accuracy_scores_train = []
cross_val_scores = []

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)


for i, fit in enumerate(fits):
    fit.fit(X_train, y_train)
    accuracy_scores_test.append(accuracy_score(y_test, fit.predict(X_test)))
    accuracy_scores_train.append(accuracy_score(y_train, fit.predict(X_train)))
    cross_val_scores.append(cross_val_score(fit, X, y, cv=skf).mean())
    if i < 5:
        y_pred_proba = fit.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{fit} AUC: {auc_score:.2f}")

plt.legend(loc='best', bbox_to_anchor=(1.0, 0.9))
plt.title("ROC curve for all models")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

accuracy_df = pd.DataFrame(
    {"test": accuracy_scores_test,
     "train": accuracy_scores_train,
      "cross_val": cross_val_scores}, index=[str(fit) for fit in fits]
      )

accuracy_df.to_csv("accuracy_scores.csv")

