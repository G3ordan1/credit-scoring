import pandas as pd
import numpy as np
# Preparation and preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# Pipeline
from sklearn.pipeline import Pipeline
# Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Performance evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
from scipy.stats import ks_2samp


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

ys[2] = ys[2].replace({1: 0, 2: 1})

# Looking for categorical and binary variables
categorical_columns = [[], [], [], []]
binary_columns = [[], [], [], []]
for i, X in enumerate(Xs):
    for j in range(X.shape[1]):
        x = X.iloc[:, j].unique()
        if len(x) < 13 and len(x) > 2:
            categorical_columns[i].append(j)
        elif len(x) == 2:
            binary_columns[i].append(j)

# Find numeric features
not_numeric_columns = [[], [], [], []]
for i in range(4):
    not_numeric_columns[i] = categorical_columns[i] + binary_columns[i]
numeric_columns_index = [[], [], [], []]
for i, X in enumerate(Xs):
    numeric_columns_index[i] = [x for x in range(
        X.shape[1]) if x not in not_numeric_columns[i]]

X0_train, X0_test, y0_train, y0_test = train_test_split(
    Xs[0], ys[0], test_size=0.2, random_state=7)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    Xs[1], ys[1], test_size=0.2, random_state=7)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    Xs[2], ys[2], test_size=0.3, random_state=7)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    Xs[3], ys[3], test_size=0.3, random_state=7)

datasets = {"Australian Credit": (X0_train, X0_test, y0_train, y0_test),
            "GMSC": (X1_train, X1_test, y1_train, y1_test),
            "German Credit": (X2_train, X2_test, y2_train, y2_test),
            "UCI Credit Card": (X3_train, X3_test, y3_train, y3_test)}


def ks_statistic(y_true, y_pred_prob):
    # Assuming y_pred_prob is the probability for the positive class
    y_true = y_true.astype(int)
    return ks_2samp(y_pred_prob[y_true == 1], y_pred_prob[y_true == 0]).statistic


ks_scorer = make_scorer(ks_statistic, response_method='predict_proba')

metrics = {'accuracy': 'accuracy', 'precision': 'precision',
           'recall': 'recall', 'f1': 'f1', 'ks': ks_scorer}

models = {"Logistic Regression": LogisticRegression(class_weight={0: 5, 1: 1}),
            "Random Forest": RandomForestClassifier(class_weight={0: 5, 1: 1}),
            "LDA": LinearDiscriminantAnalysis(),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(),
            "Decision Tree": DecisionTreeClassifier(class_weight={0: 5, 1: 1}),
            "SVM": SVC(class_weight={0: 5, 1: 1}, probability=True)}
            
# Transforming the data
ct0 = ColumnTransformer([
    ("standardised", StandardScaler(), [1]),
    ("robust", RobustScaler(), [2, 4, 6, 9, 12, 13]),
    ("binary", MinMaxScaler(), binary_columns[0]),
    ("categorical", OneHotEncoder(
        handle_unknown='ignore'), categorical_columns[0])
])

ct1 = ColumnTransformer([
    ("imputer", SimpleImputer(strategy='median'), [4, 9]),
    ("standardised", StandardScaler(), ["age"]),
    ("robust", RobustScaler(), [0, 2, 3, 4, 5, 6, 7, 8, 9])
])

ct2 = ColumnTransformer([
    ("standardised", StandardScaler(), ["Age"]),
    ("robust", RobustScaler(), [1, 4]),
    ("binary", MinMaxScaler(), binary_columns[2]),
    ("categorical", OneHotEncoder(), categorical_columns[2])
])

ct3 = ColumnTransformer([
    ("standardised", StandardScaler(), ["AGE"]),
    ("robust", RobustScaler(), [1, 4]),
    ("binary", MinMaxScaler(), binary_columns[3]),
    ("categorical", OneHotEncoder(), categorical_columns[3])
])


Pipeline0 = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                     LogisticRegression(class_weight={0: 5, 1: 1}))])



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score = cross_val_score(Pipeline0, X0_train, y0_train, cv=skf)
score

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

GridSearchCV(Pipeline0, param_grid, cv=skf).fit(
    X0_train, y0_train).best_estimator_

Pipeline0.fit(X0_train, y0_train)
Pipeline0.score(X0_train, y0_train)
Pipeline0.score(X0_test, y0_test)


pd.DataFrame(cross_validate(Pipeline0, X0_train, y0_train,
             scoring=metrics, cv=skf, return_train_score=True))

# AC pipelines
Pipeline0_lr = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                     LogisticRegression(class_weight={0: 5, 1: 1}))])

Pipeline0_rf = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                        RandomForestClassifier(class_weight={0: 5, 1: 1}))])
Pipeline0_lda = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                            LinearDiscriminantAnalysis())])
Pipeline0_knn = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                            KNeighborsClassifier())])
Pipeline0_mlp = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                            MLPClassifier(max_iter=1000))])
Pipeline0_dt = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                            DecisionTreeClassifier(class_weight={0: 5, 1: 1}))])
Pipeline0_svm = Pipeline(steps=[("preprocessor", ct0), ("classifier",
                            SVC(class_weight={0: 5, 1: 1}, probability=True))])

# GMSC pipelines
Pipeline1_lr = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                       LogisticRegression(class_weight={0: 5, 1: 1}))])
Pipeline1_rf = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                       RandomForestClassifier(class_weight={0: 5, 1: 1}))])
Pipeline1_lda = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                        LinearDiscriminantAnalysis())])
Pipeline1_knn = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                        KNeighborsClassifier())])
Pipeline1_mlp = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                        MLPClassifier(max_iter=1000))])
Pipeline1_dt = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                       DecisionTreeClassifier(class_weight={0: 5, 1: 1}))])
Pipeline1_svm = Pipeline(steps=[("preprocessor", ct1), ("classifier",
                                                        SVC(class_weight={0: 5, 1: 1}, probability=True))])

# German Credit pipelines
Pipeline2_lr = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                       LogisticRegression(class_weight={0: 5, 1: 1}))])
Pipeline2_rf = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                       RandomForestClassifier(class_weight={0: 5, 1: 1}))])
Pipeline2_lda = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                        LinearDiscriminantAnalysis())])
Pipeline2_knn = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                        KNeighborsClassifier())])
Pipeline2_mlp = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                        MLPClassifier(max_iter=1000))])
Pipeline2_dt = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                       DecisionTreeClassifier(class_weight={0: 5, 1: 1}))])
Pipeline2_svm = Pipeline(steps=[("preprocessor", ct2), ("classifier",
                                                        SVC(class_weight={0: 5, 1: 1}, probability=True))])

# UCI Credit Card pipelines
Pipeline3_lr = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                       LogisticRegression(class_weight={0: 5, 1: 1}))])
Pipeline3_rf = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                       RandomForestClassifier(class_weight={0: 5, 1: 1}))])
Pipeline3_lda = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                        LinearDiscriminantAnalysis())])
Pipeline3_knn = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                        KNeighborsClassifier())])
Pipeline3_mlp = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                        MLPClassifier(max_iter=1000))])
Pipeline3_dt = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                       DecisionTreeClassifier(class_weight={0: 5, 1: 1}))])
Pipeline3_svm = Pipeline(steps=[("preprocessor", ct3), ("classifier",
                                                        SVC(class_weight={0: 5, 1: 1}, probability=True))])

# Cross validation
ac_pipelines = [Pipeline0_lr, Pipeline0_rf, Pipeline0_lda, Pipeline0_knn,
                Pipeline0_mlp, Pipeline0_dt, Pipeline0_svm]
gmsc_pipelines = [Pipeline1_lr, Pipeline1_rf, Pipeline1_lda, Pipeline1_knn,
                  Pipeline1_mlp, Pipeline1_dt, Pipeline1_svm]
german_credit_pipelines = [Pipeline2_lr, Pipeline2_rf, Pipeline2_lda, Pipeline2_knn,
                           Pipeline2_mlp, Pipeline2_dt, Pipeline2_svm]
uci_credit_card_pipelines = [Pipeline3_lr, Pipeline3_rf, Pipeline3_lda, Pipeline3_knn,
                             Pipeline3_mlp, Pipeline3_dt, Pipeline3_svm]


def cv_pipelines(pipelines, X_train, y_train, metrics, cv):
    results = []
    for pipeline in pipelines:
        result = cross_validate(pipeline, X_train, y_train, scoring=metrics, cv=cv, return_train_score=True)
        results.append(pd.DataFrame(result))
    return results

def test_pipeline(pipelines, X_train, y_train, X_test, y_test):
    results = []
    for pipeline in pipelines:
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        results.append({"train_score": train_score, "test_score": test_score})
    return results

cv_pipelines(ac_pipelines, X0_train, y0_train, metrics, skf)