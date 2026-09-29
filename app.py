import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {font-size: 36px; font-weight: bold; color: #2E86C1; text-align: center; margin-bottom: 20px;}
    .card {background-color: #F4F6F7; padding: 20px; border-radius: 12px; text-align: center; margin: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);}
    .card h2 {margin: 0; font-size: 28px; color: #1B4F72;}
    .card p {margin: 5px 0; font-size: 14px; color: #7D3C98;}
    .positive {color: green; font-weight: bold;}
    .negative {color: red; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Advanced Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("Run Prediction")

# Multi-ticker comparison
multi_tickers = st.sidebar.multiselect("Compare Multiple Tickers", ["AAPL","TSLA","MSFT"], default=["AAPL","TSLA","MSFT"])
run_comparison = st.sidebar.button("Run Comparison")

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

def predict_stock(data):
    X = data[['Prev_Close','MA10','MA20','Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    prev_date = data.index[-1].strftime("%Y-%m-%d")
    prev_close = data['Close'].iloc[-1].item()
    next_day = model.predict([X.iloc[-1].values])[0].item()
    next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    change = ((next_day - prev_close) / prev_close) * 100
    r2 = model.score(X_test, y_test)

    return prev_date, prev_close, next_date, next_day, change, r2

if run_prediction:
    data = get_stock_data(ticker, period)
    if data is None or data.empty:
        st.error("❌ Invalid ticker or no data available.")
    else:
        prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)

        # Styled cards
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'><h2>Previous Close</h2><p>{prev_date}</p><h2>${prev_close:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><h2>Predicted Next Close</h2><p>{next_date}</p><h2>${next_day:.2f}</h2><p class='{ 'positive' if change>0 else 'negative' }'>{change:.2f}%</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><h2>Model R² Score</h2><p>Performance Metric</p><h2>{r2:.3f}</h2></div>", unsafe_allow_html=True)

        # ✅ Download button
        export_df = pd.DataFrame({
            "Date":[prev_date,next_date],
            "Close":[prev_close,next_day],
            "Change%":[0,change]
        })
        st.download_button("📥 Download Predictions (CSV)", export_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        st.download_button("📥 Download Predictions (Excel)", export_df.to_excel(index=False, engine="openpyxl"), "predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if run_comparison:
    results = []
    for t in multi_tickers:
        data = get_stock_data(t, period)
        if data is not None and not data.empty:
            prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)
            results.append([t, prev_date, prev_close, next_date, next_day, change, r2])

    if results:
        comp_df = pd.DataFrame(results, columns=["Ticker","Prev Date","Prev Close","Next Date","Predicted Close","Change%","R²"])
        st.subheader("📊 Multi‑Ticker Comparison")
        st.dataframe(comp_df)
        st.download_button("📥 Download Comparison (CSV)", comp_df.to_csv(index=False).encode("utf-8"), "comparison.csv", "text/csv")
