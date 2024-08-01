# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter=' ', header=None)

    # Assign column names (from the dataset documentation)
    columns = ['Status', 'Duration', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
               'Personal_status', 'Debtors', 'Residence_since', 'Property', 'Age', 'Installment_plans', 'Housing', 
               'Number_credits', 'Job', 'Liable', 'Telephone', 'Foreign_worker', 'Target']
    data.columns = columns

    # Encode categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    # Normalize numerical variables
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include='number').columns.difference(['Target'])
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Split data into training and testing sets
    X = data.drop('Target', axis=1)
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
