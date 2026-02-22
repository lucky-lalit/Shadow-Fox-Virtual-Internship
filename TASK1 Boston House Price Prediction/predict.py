import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def predict_price():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_housing_model.joblib")
    csv_path = os.path.join(base_dir, "HousingData.csv")
    
    # load model
    model = joblib.load(model_path)
    
    df = pd.read_csv(csv_path)
    # clean nulls
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('MEDV', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    
    # feature names
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    
    input_data = []
    med = X.median().to_dict()
    
    # sample values for demo
    sample = med.copy()
    sample['RM'] = 6.5
    sample['LSTAT'] = 5.0 
    
    print("\nPredicting with sample:")
    for f in features:
        val = sample[f]
        print(f"  {f}: {val}")
        input_data.append(val)
        
    input_arr = np.array(input_data).reshape(1, -1)
    scaled = scaler.transform(input_arr)
    pred = model.predict(scaled)[0]
    
    print(f"Prediction: ${pred * 1000:,.2f}")

if __name__ == "__main__":
    predict_price()
