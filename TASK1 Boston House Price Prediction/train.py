import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from preprocess import preprocess_data

def fine_tune():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "HousingData.csv")
    X_train, X_test, y_train, y_test, features = preprocess_data(csv_path)
    
    # hyperparameters for tuning
    param_grid = {
        'n_estimators': [10, 20, 30],
        'max_depth': [5, 8, None],
        'min_samples_split': [2, 5]
    }
    
    print("tuning model...")
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    print(f"best params: {grid.best_params_}")
    
    # eval
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # save
    model_dir = os.path.join(base_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(best_rf, os.path.join(model_dir, "best_housing_model.joblib"))
    print("model saved.")
    
    return best_rf

if __name__ == "__main__":
    fine_tune()
