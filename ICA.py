# IMPORT DATA MANIPULATION AND ANALYSIS LIBRARIES
import pandas as pd
import numpy as np

# IMPORT DATA VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT MACHINE LEARNING LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

# IMPORT MODEL EVALUATION AND HYPERPARAMETER TUNING
from sklearn.model_selection import cross_val_score, GridSearchCV

# import dataset
mydf = pd.read_csv("heart.csv")
print(mydf)

print(mydf.describe())

print("\nFirst Few Rows in the Datset:")
print(mydf.head())

print("\nLast Few Rows in the Datset:")
print(mydf.tail())


print("\nChecking for Missing Values in the Datset:")
print(mydf.isnull().sum())


# Visualise the distributioof numerical features
# age distribution of patients
plt.figure(figsize=(12, 8))
sns.histplot(data=mydf, x="age", kde=True)
plt.title(" Age Distribution of Patients")
plt.show()

# Corellation Matrix Plot
plt.figure(figsize=(10, 8))
sns.heatmap(mydf.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()
