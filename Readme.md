# Credit Scoring Model Benchmarking

This repository contains the full code and accompanying dissertation for my final-year project:  
**Classifying Credit Score Using Different Models and Evaluating Their Performance** - September 2024

The objective is to compare several machine learning and statistical models on multiple credit scoring datasets, using consistent pre-processing, cross-validation, and performance metrics.

## 🔍 Project Summary

Credit scoring is critical for financial institutions, as poor credit decisions can lead to loss of revenue or even systemic risk. This project evaluates 7 different classification models on 4 publicly available credit datasets to identify which models perform best under various conditions (e.g., class imbalance, dataset size).

Key points:
- Models evaluated: Logistic Regression, Linear Discriminant Analysis, Decision Tree, K-Nearest Neighbors, Support Vector Machines, Random Forest, Multi-layer Perceptron.
- Datasets used: German Credit, Australian Credit, Taiwan Credit, Give Me Some Credit (Kaggle).
- Techniques: Stratified Cross-Validation, Hyperparameter Tuning (`RandomizedSearchCV`, `GridSearchCV`), Evaluation with 7 metrics (Accuracy, Precision, Recall, Cohen’s Kappa, AUC, KS, etc.)
- Code written in **Python 3.11** using `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## 📈 Results

| Dataset         | Best Performing Model |
|----------------|------------------------|
| Australian      | Random Forest          |
| GMSC (Kaggle)   | Random Forest          |
| German Credit   | Random Forest / SVM    |
| Taiwan Credit   | Random Forest          |

Random Forest consistently outperformed others, offering strong accuracy, robustness to class imbalance, and good AUC scores. Decision Trees performed the worst across all datasets, confirming the benefit of ensemble methods.

## 🧠 Tech Stack

- Python 3.11
- scikit-learn 1.5.1
- pandas 2.2.2
- numpy 1.23.5
- matplotlib 3.8.4

## 📁 Structure
├── main.pdf                   # Final dissertation (A-grade, proudly LaTeX'd)  
├── notebooks/  
│   ├── Eda.ipynb           # Exploratory Data Analysis  
│   ├── Tuning.ipynb        # Hyperparameter tuning with RandomizedSearchCV  
│   └── final_test.ipynb    # Final model evaluation with cross-validation  
├── datasets/                  
├── README.md

## 📚 Datasets

1. **German Credit** (UCI Repository)
2. **Australian Credit** (UCI Repository)
3. **Taiwan Credit Card** (UCI Repository)
4. **Give Me Some Credit** (Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit)

