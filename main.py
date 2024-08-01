import pandas as pd
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_preprocess_data
from src.exploratory_data_analysis import plot_data_distribution, plot_correlation_matrix
from src.feature_engineering import select_features
from src.model_building import train_logistic_regression, train_decision_tree, train_random_forest
from src.model_evaluation import evaluate_model
from src.model_interpretation import get_feature_importances


# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/german.data')

# Exploratory Data Analysis (EDA)
data = pd.read_csv('data/german.data', delimiter=' ', header=None)
plot_data_distribution(data, 4)  # Example for 'Credit_amount' column
plot_correlation_matrix(data)

# Feature Engineering
X_new, selected_features = select_features(X_train, y_train)

# Model Building
lr_model = train_logistic_regression(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Model Evaluation
print('Logistic Regression:')
lr_accuracy, lr_report, lr_auc = evaluate_model(lr_model, X_test, y_test)
print(lr_report)

print('Decision Tree:')
dt_accuracy, dt_report, dt_auc = evaluate_model(dt_model, X_test, y_test)
print(dt_report)

print('Random Forest:')
rf_accuracy, rf_report, rf_auc = evaluate_model(rf_model, X_test, y_test)
print(rf_report)

# Model Interpretation
feature_importance_df = get_feature_importances(rf_model, X_train.columns)
print(feature_importance_df)

# Pie Chart Visualization
labels = ['Logistic Regression', 'Decision Tree', 'Random Forest']
sizes = [lr_accuracy, dt_accuracy, rf_accuracy]  # Use accuracies or any other metric
colors = ['gold', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0)  # explode the 1st slice (Logistic Regression)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Model Accuracy Comparison')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
