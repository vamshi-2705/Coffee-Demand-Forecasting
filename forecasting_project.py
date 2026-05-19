import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # 1. Assign Dates (Since only transaction_time is provided)
    print("Assigning dates based on time resets...")
    df['time_dt'] = pd.to_datetime(df['transaction_time'], format='mixed')
    
    # Identify where time restarts (new day)
    # We assume transaction_id is sequential and sorted by time within days
    df['time_diff'] = df['time_dt'].diff().dt.total_seconds()
    df['new_day'] = (df['time_diff'] < 0).astype(int)
    df['day_offset'] = df['new_day'].cumsum()
    
    # Start date: Jan 1st, 2025
    start_date = pd.Timestamp('2025-01-01')
    df['transaction_date'] = df['day_offset'].apply(lambda x: start_date + timedelta(days=x))
    
    # Combine date and time
    df['datetime'] = pd.to_datetime(df['transaction_date'].dt.strftime('%Y-%m-%d') + ' ' + df['transaction_time'])
    
    # 2. Calculate Revenue
    df['revenue'] = df['transaction_qty'] * df['unit_price']
    
    # 3. Hour of day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    
    print(f"Data cleaned. Total days: {df['day_offset'].max() + 1}")
    return df

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Daily Sales Trend
    plt.figure(figsize=(12, 6))
    daily_sales = df.groupby('transaction_date')['revenue'].sum()
    daily_sales.plot(color='#1f77b4', linewidth=2)
    plt.title('Daily Sales Trend (2025)', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.savefig('plots/daily_sales_trend.png')
    plt.close()

    # 2. Hourly Demand Trend
    plt.figure(figsize=(10, 6))
    hourly_sales = df.groupby('hour')['transaction_qty'].sum()
    sns.barplot(x=hourly_sales.index, y=hourly_sales.values, palette='viridis')
    plt.title('Hourly Transaction Volume', fontsize=15)
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Quantity')
    plt.savefig('plots/hourly_demand_trend.png')
    plt.close()

    # 3. Store-wise Revenue
    plt.figure(figsize=(10, 6))
    store_rev = df.groupby('store_location')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=store_rev.values, y=store_rev.index, palette='magma')
    plt.title('Revenue by Store Location', fontsize=15)
    plt.savefig('plots/store_wise_revenue.png')
    plt.close()

    # 4. Product Category Sales
    plt.figure(figsize=(10, 6))
    cat_sales = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
    cat_sales.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Revenue by Product Category', fontsize=15)
    plt.ylabel('')
    plt.savefig('plots/product_category_sales.png')
    plt.close()

    # 5. Peak Hours Heatmap
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(index='day_of_week', columns='hour', values='transaction_qty', aggfunc='sum')
    # Sort days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days)
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=False)
    plt.title('Peak Demand Heatmap (Day vs Hour)', fontsize=15)
    plt.savefig('plots/peak_hours_heatmap.png')
    plt.close()

    # 6. Top Selling Products
    plt.figure(figsize=(10, 6))
    top_prod = df.groupby('product_detail')['transaction_qty'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_prod.values, y=top_prod.index, palette='rocket')
    plt.title('Top 10 Selling Products by Quantity', fontsize=15)
    plt.savefig('plots/top_selling_products.png')
    plt.close()

    print("EDA plots saved in 'plots/' directory.")

def forecast_daily_sales(df):
    print("Starting Daily Sales Forecasting...")
    
    # Aggregate by day
    daily_data = df.groupby('transaction_date')['revenue'].sum().reset_index()
    daily_data.columns = ['ds', 'y']
    
    # Split: 80% Train, 20% Test
    train_size = int(len(daily_data) * 0.8)
    train, test = daily_data.iloc[:train_size], daily_data.iloc[train_size:]
    
    results = {}

    # 1. Naive Forecast
    test_naive = test.copy()
    test_naive['yhat'] = train.iloc[-1]['y']
    results['Naive'] = test_naive

    # 2. Moving Average (7 days)
    test_ma = test.copy()
    test_ma['yhat'] = daily_data['y'].rolling(window=7).mean().iloc[train_size-1 : len(daily_data)-1].values
    results['MovingAverage'] = test_ma

    # 3. Prophet
    print("Training Prophet model...")
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(train)
    future = m.make_future_dataframe(periods=len(test))
    forecast = m.predict(future)
    test_prophet = test.copy()
    test_prophet['yhat'] = forecast.iloc[train_size:]['yhat'].values
    results['Prophet'] = test_prophet

    # Evaluate
    metrics = []
    for name, res in results.items():
        mae = mean_absolute_error(res['y'], res['yhat'])
        rmse = np.sqrt(mean_squared_error(res['y'], res['yhat']))
        mape = np.mean(np.abs((res['y'] - res['yhat']) / res['y'])) * 100
        metrics.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape})
    
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Evaluation:")
    print(metrics_df)
    metrics_df.to_csv('model_evaluation.csv', index=False)
    
    # Plot Prophet Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train['ds'], train['y'], label='Train')
    plt.plot(test['ds'], test['y'], label='Test')
    plt.plot(test_prophet['ds'], test_prophet['yhat'], label='Prophet Forecast', linestyle='--')
    plt.title('Daily Revenue Forecast (Prophet)', fontsize=15)
    plt.legend()
    plt.savefig('plots/prophet_forecast_daily.png')
    plt.close()
    
    return results, metrics_df

def main():
    df = load_and_clean_data('dataset.csv')
    perform_eda(df)
    results, metrics = forecast_daily_sales(df)
    
    # Export final dataset for Streamlit
    df.to_csv('processed_data.csv', index=False)
    print("Pipeline complete. Processed data saved to 'processed_data.csv'")

if __name__ == "__main__":
    main()
