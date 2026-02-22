import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from preprocess import preprocess_data

def plot_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "HousingData.csv")
    model_path = os.path.join(base_dir, "models", "best_housing_model.joblib")
    out_dir = os.path.join(base_dir, "visuals")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    X_train, X_test, y_train, y_test, feat_names = preprocess_data(csv_path)
    model = joblib.load(model_path)
    
    feat_names = [f for f in feat_names if f != 'MEDV']
    
    # features
    importances = model.feature_importances_
    idx = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title("Importance")
    plt.barh(range(len(idx)), importances[idx], color='blue')
    plt.yticks(range(len(idx)), [feat_names[i] for i in idx])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance.png"))
    
    # pred vs actual
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
    plt.title('Actual vs Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "results.png"))
    
    return os.path.join(out_dir, "importance.png"), os.path.join(out_dir, "results.png")

if __name__ == "__main__":
    plot_results()
