import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

# Page setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .card {
        background-color: #F4F6F7;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .card h2 {
        margin: 0;
        font-size: 28px;
        color: #1B4F72;
    }
    .card p {
        margin: 5px 0;
        font-size: 14px;
        color: #7D3C98;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Advanced Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)

def get_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        data['Prev_Close'] = data['Close'].shift(1)
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data = data.dropna()
        return data
    except Exception:
        return None

if st.sidebar.button("Run Prediction"):
    data = get_stock_data(ticker, period)

    if data is None or data.empty:
        st.error("❌ Invalid ticker or no data available.")
    else:
        # Train model
        X = data[['Prev_Close','MA10','MA20','Volume']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Previous and next day info
        prev_date = data.index[-1].strftime("%Y-%m-%d")
        prev_close = data['Close'].iloc[-1]
        next_day = model.predict([X.iloc[-1].values])[0].item()  # ✅ safe scalar conversion
        next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        change = ((next_day - prev_close) / prev_close) * 100

        # Display results in styled cards
        col1, col2, col3 = st.columns(3)

