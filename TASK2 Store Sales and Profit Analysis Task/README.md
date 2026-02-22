# Store Sales and Profit Analysis

## Project Overview
This project analyzes the performance of a retail store using the "Sample - Superstore" dataset. The goal is to provide insights into sales trends, profitability, and product performance to help drive strategic business decisions.

## How to Run the Project

### 1. Requirements
Make sure you have Python installed. You will also need to install the following libraries:
```bash
pip install pandas plotly scikit-learn
```

### 2. Running the Analysis
Simply run the `analysis.py` script using the following command:
```bash
python analysis.py
```

### 3. Viewing the Results
After running the script, an `assets/` folder will be created (if it doesn't already exist) containing interactive visualizations:
- `monthly_sales.html`: Line chart showing sales over time.
- `category_sales.html`: Bar chart comparing sales across categories.
- `segment_profit.html`: Pie chart showing profit contribution by customer segment.
- `subcat_profit.html`: Breakdown of profit by sub-category.
- `sales_forecast.html`: A 6-month sales forecast model.

## Features
- **Data Cleaning**: Handled date formats and missing values.
- **Trend Analysis**: Insights into seasonal sales patterns.
- **Profitability Analysis**: Identification of high-performing segments and problematic sub-categories.
- **Machine Learning**: A simple Linear Regression model for sales forecasting.
