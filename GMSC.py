from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training = pd.read_csv("datasets/GMSC/cs-training.csv", index_col=0)
test = pd.read_csv("datasets/GMSC/cs-test.csv", index_col=0)
training.head()
X_train = training.drop('SeriousDlqin2yrs', axis=1)
y_train = training['SeriousDlqin2yrs']

X_test = test.drop('SeriousDlqin2yrs', axis=1)
y_test = test['SeriousDlqin2yrs']

X_train.head()

X_train.isnull().sum()
training.MonthlyIncome.mean()
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
np.isnan(X_train).sum()

sns.countplot(y_train)
y_train.value_counts()
sns.barplot(x=["No Dlq", "Dlq"], y=y_train.value_counts().values).get_label()

sns.countplot(x=[1, 1, 2, 1], hue=[1, 1, 2, 1]).set_ylabel("Frequency")
plt.show()
flights = sns.load_dataset("flights")
flights_wide = flights.pivot(
    index="year", columns="month", values="passengers")
flights_wide.shape
flights.shape
flights_wide.head()
flights.head()
sns.barplot(flights_wide)
plt.show()
