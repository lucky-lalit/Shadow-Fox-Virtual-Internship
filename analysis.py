import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# --- Author: [Your Name] ---
# Project: Store Sales and Profit Analysis

def main():
    # Create a folder to store our charts
    if not os.path.exists('assets'):
        os.makedirs('assets')

    print("--- Starting Store Analysis ---")

    # 1. Loading and Cleaning the Data
    print("Step 1: Loading dataset...")
    try:
        # Some CSV files require windows encoding
        data = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
    except:
        data = pd.read_csv('Sample - Superstore.csv')

    # Converting string dates to actual Python datetime objects
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Ship Date'] = pd.to_datetime(data['Ship Date'])

    # Dropping any empty rows just in case
    data = data.dropna()
    
    # Sorting everything by the order date
    data = data.sort_values('Order Date')

    # 2. Analyzing Sales Trends
    print("Step 2: Analyzing sales trends over time...")
    # We group by Month End (ME) to see the monthly total sales
    monthly_data = data.resample('ME', on='Order Date')['Sales'].sum().reset_index()
    
    sales_fig = px.line(monthly_data, x='Order Date', y='Sales', 
                        title='Monthly Sales Performance',
                        labels={'Sales': 'Total Revenue ($)', 'Order Date': 'Date'},
                        template='plotly_dark')
    sales_fig.write_html('assets/monthly_sales.html')

    # Category comparison
    category_data = data.groupby('Category')['Sales'].sum().reset_index()
    cat_fig = px.bar(category_data, x='Category', y='Sales', color='Category', 
                     title='Sales Volume by Category', template='plotly_dark')
    cat_fig.write_html('assets/category_sales.html')

    # 3. Profitability Analysis
    print("Step 3: Calculating profitability...")
    # Pie chart for segments
    segment_data = data.groupby('Segment')['Profit'].sum().reset_index()
    seg_fig = px.pie(segment_data, values='Profit', names='Segment', 
                     title='Profit Share by Customer Segment',
                     hole=.3, template='plotly_dark')
    seg_fig.write_html('assets/segment_profit.html')
    
    # Detailed sub-category profit
    subcat_data = data.groupby('Sub-Category')['Profit'].sum().reset_index().sort_values('Profit', ascending=False)
    subcat_fig = px.bar(subcat_data, x='Sub-Category', y='Profit', 
                        title='Profit/Loss by Sub-Category', template='plotly_dark')
    subcat_fig.write_html('assets/subcat_profit.html')

    # 4. Simple Sales Forecasting (Machine Learning)
    print("Step 4: Training a simple projection model...")
    # We use a simple time index (0, 1, 2...) to find the trend
    monthly_data['Months_Passed'] = np.arange(len(monthly_data))
    X = monthly_data[['Months_Passed']]
    y = monthly_data['Sales']
    
    # Linear Regression is great for showing simple growth trends
    model = LinearRegression()
    model.fit(X, y)
    
    # Let's predict the next 6 months
    future_X = np.arange(len(monthly_data), len(monthly_data) + 6).reshape(-1, 1)
    future_preds = model.predict(future_X)
    
    # Creating dates for the forecast
    last_date = monthly_data['Order Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=7, freq='ME')[1:]
    
    # Combine historical and forecast data for the chart
    forecast_df = pd.DataFrame({'Order Date': future_dates, 'Sales': future_preds, 'Status': 'Forecast'})
    historical_df = monthly_data[['Order Date', 'Sales']].copy()
    historical_df['Status'] = 'Actual'
    
    plot_df = pd.concat([historical_df, forecast_df])
    
    forecast_fig = px.line(plot_df, x='Order Date', y='Sales', color='Status', 
                          title='Sales Growth Forecast (Next 6 Months)', template='plotly_dark')
    forecast_fig.write_html('assets/sales_forecast.html')
    
    print("\n--- Analysis Complete ---")
    print(f"Trend Coefficient: {model.coef_[0]:.2f}")
    print("All charts have been saved to the 'assets/' folder.")
    print("Open README.md for instructions.")

if __name__ == "__main__":
    main()
