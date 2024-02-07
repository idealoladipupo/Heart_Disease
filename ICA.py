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
