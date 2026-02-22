import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # outlier capping using iqr
    def cap_outliers(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        return column.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    to_cap = [col for col in df.columns if col not in ['CHAS', 'MEDV']]
    for col in to_cap:
        df[col] = cap_outliers(df[col])
        
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df.columns.tolist()

if __name__ == "__main__":
    # Use relative path for portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "HousingData.csv")
    
    if os.path.exists(csv_path):
        X_train, X_test, y_train, y_test, cols = preprocess_data(csv_path)
        print("Preprocessing complete.")
        print(f"Train set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
    else:
        print(f"Error: {csv_path} not found.")
