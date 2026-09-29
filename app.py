import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("Run Prediction")   # ✅ store button state

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

if run_prediction:   # ✅ check button state
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
        next_day = model.predict([X.iloc[-1].values])[0].item()
        next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        change = ((next_day - prev_close) / prev_close) * 100

        # Styled cards
        col1, col2, col3 = st.columns(3)
        col1.success(f"Previous Close ({prev_date}): ${prev_close:.2f}")
        col2.success(f"Predicted Next Close ({next_date}): ${next_day:.2f} ({change:.2f}%)")
        col3.info(f"Model R² Score: {model.score(X_test, y_test):.3f}")

        # Interactive chart
        fig = px.line(data, x=data.index, y=["Close","MA10","MA20"],
                      labels={"value":"Price","index":"Date"},
                      title=f"{ticker} Stock Price & Moving Averages")
        st.plotly_chart(fig, use_container_width=True)
