# Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters

An internship project focused on time-series analysis, sales forecasting, and operational optimization.

## Project Structure
- `forecasting_project.py`: Main processing and modeling script.
- `app.py`: Streamlit dashboard for interactive visualization.
- `dataset.csv`: Raw transaction data.
- `processed_data.csv`: Cleaned data with synthesized dates for analysis.
- `plots/`: Directory containing EDA and forecast visualizations.
- `requirements.txt`: Project dependencies.
- `report.md`: Full research-style report.
- `ppt_content.md`: Content for presentation slides.
- `executive_summary.txt`: One-page summary for stakeholders.

## Installation
1. Clone the repository/folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Run Data Pipeline
Executes data cleaning, EDA, and model training:
```bash
python forecasting_project.py
```

### 2. Launch Streamlit Dashboard
Start the interactive dashboard:
```bash
streamlit run app.py
```

## Forecasting Methodology
- **Data Synthesis**: Since raw data only contained time but no dates, dates were inferred by detecting time-resets in sequential transactions.
- **Models Used**: Naive Forecast, Moving Average, and Prophet (Time-Series).
- **KPIs**: Mean Absolute Error (MAE), RMSE, and Forecast Accuracy (%).

## Key Visualizations Included
- Daily Sales Trends
- Hourly Demand Heatmaps
- Store-wise Revenue Analysis
- Product Category Distribution
- Prophet Forecast Projections
