from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

gc = fetch_ucirepo(id=144)
X = gc.data.features
X.iloc[:, 8] = X.iloc[:, 8].replace({"A91": "M", "A92": "F", "A93": "M", "A94": "M", "A95": "F"})
X.columns = ["Account_status","Duration","Credit_history","Purpose","Credit_amount","Savings_bonds","Present_employment_since","Installment_rate", "Gender","Other_debtors_guarantors","Resident_since","Property","Age","Other_installment_plans","Housing","Existing_credits","Job","People_maintenance_for","Telephone","Foreign_worker"]
y = gc.data.targets
y.columns = ["Credit_risk"]
dataset = pd.DataFrame.join(X, y)
dataset.to_csv("datasets/german_credit.csv", index=False)


statlog_australian_credit_approval = fetch_ucirepo(id=143)
X = statlog_australian_credit_approval.data.features
y = statlog_australian_credit_approval.data.targets
dataset = pd.DataFrame.join(X, y)
dataset.to_csv("datasets/australian_credit.csv", index=False)

print(statlog_australian_credit_approval.variables)