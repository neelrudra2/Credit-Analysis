# src/model_evaluation.py
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, report, auc
