import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Afficionado Coffee Dashboard", layout="wide")

st.title("☕ Data-Driven Forecasting & Peak Demand Prediction")
st.markdown("### Afficionado Coffee Roasters - Internship Project")

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists('processed_data.csv'):
        return None
    df = pd.read_csv('processed_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date
    return df

df = load_data()

if df is None:
    st.error("Data not found! Please run 'python forecasting_project.py' first to process the data.")
else:
    # Sidebar Filters
    st.sidebar.header("Filters")
    store = st.sidebar.multiselect("Select Store Location", options=df['store_location'].unique(), default=df['store_location'].unique())
    category = st.sidebar.multiselect("Select Product Category", options=df['product_category'].unique(), default=df['product_category'].unique())
    
    date_range = st.sidebar.date_input("Select Date Range", 
                                      [df['transaction_date'].min(), df['transaction_date'].max()],
                                      min_value=df['transaction_date'].min(),
                                      max_value=df['transaction_date'].max())

    # Filter Data
    filtered_df = df[
        (df['store_location'].isin(store)) & 
        (df['product_category'].isin(category)) &
        (df['transaction_date'] >= date_range[0]) & 
        (df['transaction_date'] <= date_range[1])
    ]

    # KPI Layout
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_revenue = filtered_df['revenue'].sum()
    total_qty = filtered_df['transaction_qty'].sum()
    avg_order = total_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
    total_transactions = len(filtered_df)

    kpi1.metric("Total Revenue", f"${total_revenue:,.2f}")
    kpi2.metric("Total Transactions", f"{total_transactions:,}")
    kpi3.metric("Avg Order Value", f"${avg_order:.2f}")
    kpi4.metric("Total Qty Sold", f"{total_qty:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue vs Quantity Toggle")
        metric_toggle = st.radio("Display Metric", ["Revenue", "Quantity"], horizontal=True)
        metric_col = 'revenue' if metric_toggle == "Revenue" else 'transaction_qty'
        
        daily_trend = filtered_df.groupby('transaction_date')[metric_col].sum().reset_index()
        fig_trend = px.line(daily_trend, x='transaction_date', y=metric_col, title=f"Daily {metric_toggle} Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("Peak Demand Heatmap")
        pivot_data = filtered_df.groupby(['day_of_week', 'hour'])['transaction_qty'].sum().reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_heat = px.density_heatmap(pivot_data, x='hour', y='day_of_week', z='transaction_qty',
                                     category_orders={'day_of_week': days},
                                     color_continuous_scale="Viridis",
                                     title="Transaction Volume by Hour & Day")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Store-wise Performance")
        store_perf = filtered_df.groupby('store_location')['revenue'].sum().sort_values().reset_index()
        fig_store = px.bar(store_perf, x='revenue', y='store_location', orientation='h', title="Revenue by Store")
        st.plotly_chart(fig_store, use_container_width=True)

    with col4:
        st.subheader("Forecast Horizon")
        if os.path.exists('model_evaluation.csv'):
            metrics = pd.read_csv('model_evaluation.csv')
            st.table(metrics)
        else:
            st.info("Run modeling to see forecast metrics.")

    st.markdown("---")
    st.subheader("Top Products")
    top_p = filtered_df.groupby('product_detail')['transaction_qty'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_prod = px.bar(top_p, x='transaction_qty', y='product_detail', color='transaction_qty')
    st.plotly_chart(fig_prod, use_container_width=True)

    # Download Button
    st.sidebar.markdown("### Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Forecast CSV", data=csv, file_name="forecast_export.csv", mime="text/csv")
