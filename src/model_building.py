# src/model_building.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    return dt_model

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model
