# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, seaborn as sb

from ucimlrepo import fetch_ucirepo 

# fetch dataset 
statlog_australian_credit_approval = fetch_ucirepo(id=143) 
  
# data (as pandas dataframes) 
X = statlog_australian_credit_approval.data.features 
y = statlog_australian_credit_approval.data.targets 
dataset = pd.DataFrame.join(X, y)

# metadata 
print(statlog_australian_credit_approval.metadata) 
  
# variable information 
print(statlog_australian_credit_approval.variables) 

# Importing the dataset
# dataset = pd.read_csv('credit_approval.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values

# One Hot Encoding the columns: 3, 4, 5, 11
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
        transformers = [('one_hot_encoder', OneHotEncoder(categories = 'auto'),[3, 4, 5, 11])],
        remainder = 'passthrough')

X = ct.fit_transform(X)

# Scaling - Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# Splitting dataset into train & test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Building the Decision Tree Classifier model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 170, random_state = 0)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("\nMean of the Accuracies after cross-validation: ", accuracies.mean())
print("\nStandard Deviation within the accuracies: ", accuracies.std())

print('\nAccuracy: ', accuracy)

# Histograms for eaach of the columns
feature_dict = {i:label for i,label in zip(
                range(14), dataset.columns)}

label_dict = {0: 'Rejected', 1: 'Approved'}
import math

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10,10))

for ax,cnt in zip(axes.ravel(), range(14)):

    # set bin sizes
    min_b = math.floor(np.min(X[:,cnt]))
    max_b = math.ceil(np.max(X[:,cnt]))
    bins = np.linspace(min_b, max_b, 25)

    # plotting the histograms
    for lab,col in zip(range(0,2), ('red', 'blue')):
        ax.hist(X[y==lab, cnt],
                   color=col,
                   label='class %s' %label_dict[lab],
                   bins=bins,
                   alpha=0.5,)
    ylims = ax.get_ylim()

    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title('Credit histogram #%s' %str(cnt+1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

fig.tight_layout()
plt.show()

pred = list()
reject_count = approved_count = 0
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        pred.append('Rejections')
    else:
        pred.append('Approvals')
pred = pd.DataFrame(pred)
pred.columns = ['Decisions']
# Visualization of Decision Counts
plt.Figure(figsize = (8, 8))
sb.set_style('darkgrid')
sb.countplot(pred['Decisions'], data = pred, edgecolor = 'black', linewidth=1.5, palette = 'dark')
plt.title('Predicted Credit Approvals')
plt.xlabel('Approval Decision')
plt.ylabel('Count')
plt.show()