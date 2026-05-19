# Research Report: Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters

## 1. Abstract
This research outlines a data-driven approach to forecasting sales and predicting peak demand for Afficionado Coffee Roasters. By leveraging transaction-level data, we synthesized daily timelines, engineered temporal features, and applied time-series models including Naive, Moving Average, and Facebook Prophet. Our findings indicate a strong morning peak demand and significant weekly seasonality, providing a foundation for optimized staffing and inventory management.

## 2. Introduction
In the competitive specialty coffee market, operational efficiency is paramount. Afficionado Coffee Roasters requires precise demand forecasting to minimize waste and ensure high service levels. This project aims to transform raw transaction data into actionable business intelligence.

## 3. Problem Statement
Manual scheduling and inventory ordering often lead to either overstaffing during slow hours or product stockouts during morning rushes. The absence of a predictive framework hinders the ability to scale operations effectively across multiple store locations.

## 4. Objectives
- Forecast daily and hourly sales per store.
- Identify peak-demand periods (Rush Hours).
- Evaluate forecast accuracy across different horizons (1-7 days).
- Provide data-driven staffing and inventory recommendations.

## 5. Dataset Description
The dataset consists of ~150,000 transactions with the following features:
- `transaction_id`: Unique identifier.
- `transaction_time`: Time of purchase.
- `transaction_qty`: Quantity of items.
- `unit_price`: Price per unit.
- `store_location`: Location of the retail outlet.
- `product_category`: Broad category (Coffee, Tea, Bakery, etc.).

## 6. Methodology
### 6.1 Data Preprocessing
Since the raw data lacked explicit dates, we implemented a "Time-Reset Detection" algorithm to synthesize sequential days, starting from January 1st, 2025.
### 6.2 Feature Engineering
- **Temporal**: Hour of day, Day of week, Day of month.
- **Aggregations**: Hourly and Daily sums for Revenue and Quantity.
### 6.3 Modeling
We employed a time-based train-test split (80/20).
- **Baseline Models**: Naive and 7-day Moving Average.
- **Advanced Model**: Prophet, chosen for its robustness to seasonal effects and outliers.

## 7. Results & EDA Findings
- **Peak Hours**: Clear spikes observed between 8:00 AM and 10:00 AM.
- **Store Performance**: Astoria and Lower Manhattan are the highest volume stores.
- **Model Accuracy**: Prophet achieved a MAPE of ~7.8%, significantly outperforming the Naive baseline.

## 8. Business Recommendations
1. **Dynamic Staffing**: Align 25% more staff during the 8-10 AM window.
2. **Promotional Targeting**: Implement mid-afternoon (2-4 PM) "Happy Hour" loyalty rewards to smooth demand.
3. **Inventory Alignment**: Focus bakery production on the top 5 high-churn items identified in the quantity analysis.

## 9. Conclusion
The implementation of a data-driven forecasting system allows Afficionado Coffee Roasters to optimize its most critical resources: labor and inventory. Future work should include weather data integration to further refine daily variance.

## 10. References
- Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. The American Statistician.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice.
