

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load model and feature columns
model = joblib.load('tuned_random_forest_model.pkl')
trained_features = joblib.load('model_features.pkl')

# Page settings
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("🧠 Demand Forecasting Dashboard")
st.markdown("This dashboard predicts **Number of Products Sold** using your trained Random Forest model.")

# File uploader
st.header("📤 Upload Your CSV Data")
uploaded_file = st.file_uploader("Upload a CSV file with feature data (no target column)", type="csv")
if uploaded_file is not None:
    # Read and preview
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(data.head())

    # Remove target column if it exists
    if 'Number of products sold' in data.columns:
        data = data.drop(columns=['Number of products sold'])

    # One-hot encode the uploaded data
    data_encoded = pd.get_dummies(data)

    # Reindex to match training features (missing columns will be filled with 0)
    data_encoded = data_encoded.reindex(columns=trained_features, fill_value=0)

    # Prediction button
    if st.button("🔮 Predict"):
        try:
            predictions = model.predict(data_encoded)
            data['Predicted Sales'] = predictions
            st.success("✅ Predictions generated successfully!")
            st.subheader("📈 Predicted Results")
            st.dataframe(data[['Predicted Sales']].head())
             # Download button
            st.download_button(
                label="📥 Download Predictions",
                data=data.to_csv(index=False),
                file_name="predicted_sales.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# Show model evaluation plots
st.header("📊 Model Performance Plots")
st.image("rmse_comparison_plot.png", caption="RMSE Comparison: Before vs. After Tuning", use_column_width=True)
st.image("r2_comparison_plot.png", caption="R² Score Comparison: Before vs. After Tuning", use_column_width=True)

st.markdown("---")
st.markdown("🔧 Built by Jianlumei Kamei | Powered by Streamlit")

