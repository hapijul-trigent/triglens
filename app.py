import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.feature_pipeline import feature_engineering
from src.inference import inference_batch
from lightgbm import LGBMClassifier  # Example: Replace with your actual model if needed

# Set page configuration for a wider layout
st.set_page_config(page_title="Triglens Dashboard", layout="wide")

# Generate Dummy Data
@st.cache_resource(show_spinner=False)
def analyze(num_records=500):
    np.random.seed(0)
    # Load the data from the CSV file in chunks
    chunks = pd.read_csv(r"C:\Users\hapijul_h\spaces\triglens\data\raw\PS_log.csv", chunksize=10000)

    # Process each chunk and concatenate results
    processed_chunks = []
    for chunk in chunks:
        processed_data = feature_engineering(chunk)
        processed_chunk = inference_batch(processed_data)
        processed_chunks.append(processed_chunk)

    # Concatenate all processed chunks into a single DataFrame
    data_with_predictions = pd.concat(processed_chunks, ignore_index=True)

    # Display the DataFrame with added columns
    del chunks, processed_chunks
    return data_with_predictions

# Load the data and display it in the dashboard
data = analyze()

# Title and Introduction
st.title("Fraud Detection Dashboard")
st.markdown("""
Welcome to the Fraud Detection Dashboard. This interface allows you to monitor transactions, analyze fraud scores, and visualize patterns for enhanced fraud detection.
""")

# KPI Section
with st.container():
    st.markdown("### Key Performance Indicators")
    kpi1, kpi2, kpi3 = st.columns([1, 1, 1])
    kpi1.metric("Total Transactions", f"{len(data)}")
    kpi2.metric("High-Risk Transactions", f"{data[data['Risk Level'] == 'High'].shape[0]}")

# Real-Time Monitoring and High-Risk Alerts
with st.container():
    st.markdown("### Transaction Monitoring")
    left_column, right_column = st.columns([2, 1])

    with left_column:
        st.subheader("Transaction Data")
        st.dataframe(data[[
            "amount", "balanceChangeOrig", "balanceChangeDest",
            "amountToOldBalanceOrigRatio", "amountToOldBalanceDestRatio",
            "transactionCountOrig", "transactionCountDest", "fraud_score", "Risk Level"
        ]].head(100), height=300)

    with right_column:
        
        high_risk_data = data[data["Risk Level"] == "High"]
        st.subheader(f"High-Risk Alerts: {high_risk_data.shape[0]}")
        st.dataframe(high_risk_data.head(500), height=300, use_container_width=True)

# Data Visualizations
with st.container():
    st.markdown("### Data Visualizations")
    col1, col2, col3 = st.columns(3)

    # Fraud Score Distribution
    with col1:
        st.subheader("Fraud Score Distribution")
        fig_score_dist = px.histogram(data, x="fraud_score", nbins=20, title="Fraud Score Distribution")
        st.plotly_chart(fig_score_dist, use_container_width=True)

    # Risk Level Breakdown
    with col2:
        st.subheader("Risk Level Breakdown")
        risk_count = data["Risk Level"].value_counts()
        fig_risk_level = px.pie(values=risk_count.values, names=risk_count.index, title="Risk Level Breakdown")
        st.plotly_chart(fig_risk_level, use_container_width=True)

    # Balance Change Analysis
    with col3:
        st.subheader("Balance Change Analysis")
        fig_balance_change = px.scatter(
            data, x="balanceChangeOrig", y="balanceChangeDest",
            color="Risk Level", title="Balance Change (Orig vs. Dest)"
        )
        st.plotly_chart(fig_balance_change, use_container_width=True)

# Transaction Types Breakdown
with st.container():
    st.markdown("### Transaction Types Breakdown")
    type_cols = ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    transaction_type_counts = data[type_cols].sum().reset_index()
    transaction_type_counts.columns = ["Transaction Type", "Count"]
    fig_transaction_types = px.bar(transaction_type_counts, x="Transaction Type", y="Count", title="Transaction Type Counts")
    st.plotly_chart(fig_transaction_types, use_container_width=True)

# # Feature Importance Visualization
# with st.container():
#     st.markdown("### Feature Importance")

#     # Example model for demonstration - replace with your own model if needed
#     model = LGBMClassifier()
#     # Fit model on a small sample of the data
#     X_sample = data[[
#         "amount", "balanceChangeOrig", "balanceChangeDest",
#         "amountToOldBalanceOrigRatio", "amountToOldBalanceDestRatio",
#         "transactionCountOrig", "transactionCountDest", "type_CASH_OUT",
#         "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
#     ]]
#     y_sample = np.random.choice([0, 1], size=X_sample.shape[0])  # Dummy target for demonstration
#     model.fit(X_sample, y_sample)

#     # Get feature importance values
#     feature_importance = model.feature_importances_
#     features = X_sample.columns
#     feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})

#     # Plot feature importance
#     fig_feature_importance = px.bar(feature_importance_df, x="Feature", y="Importance", title="Feature Importance")
#     st.plotly_chart(fig_feature_importance, use_container_width=True)

# Feature Ratios Analysis
with st.container():
    st.markdown("### Feature Ratios Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount to Old Balance Orig Ratio")
        fig_amount_orig_ratio = px.histogram(data, x="amountToOldBalanceOrigRatio", nbins=20,
                                             color="Risk Level", title="Amount to Old Balance Orig Ratio")
        st.plotly_chart(fig_amount_orig_ratio, use_container_width=True)

    with col2:
        st.subheader("Amount to Old Balance Dest Ratio")
        fig_amount_dest_ratio = px.histogram(data, x="amountToOldBalanceDestRatio", nbins=20,
                                             color="Risk Level", title="Amount to Old Balance Dest Ratio")
        st.plotly_chart(fig_amount_dest_ratio, use_container_width=True)

# Report Generation
with st.container():
    st.markdown("### Generate Custom Report")
    # Convert DataFrame to CSV for download
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )
