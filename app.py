import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# ✅ Background image styling with dark overlay
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/VikramSinghShah/stock-price-prediction/main/background.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0,0,0,0.5);
        z-index: -1;
    }
    .main-title {font-size: 42px; font-weight: bold; color: #FFFFFF; text-align: center; margin-bottom: 25px;}
    .card {background-color: rgba(244,246,247,0.9); padding: 25px; border-radius: 12px; text-align: center; margin: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);}
    .card h2 {margin: 0; font-size: 32px; color: #1B4F72;}
    .card p {margin: 8px 0; font-size: 18px; color: #7D3C98;}
    .positive {color: green; font-weight: bold; font-size: 32px;}
    .negative {color: red; font-weight: bold; font-size: 32px;}
    .section-title {text-align: center; font-size: 30px; font-weight: bold; margin-top: 30px; margin-bottom: 20px; color:#FFFFFF;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📈 Advanced Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("Run Prediction (1-Day Ahead)")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 2, 7, 3)
run_forecast = st.sidebar.button("Run Forecast Horizon")

multi_tickers = st.sidebar.multiselect("Compare Multiple Tickers", ["AAPL","TSLA","MSFT"], default=["AAPL","TSLA","MSFT"])
run_comparison = st.sidebar.button("Run Comparison")
display_mode = st.sidebar.radio("Show in Comparison", ["Table Only","Charts Only","Table + Charts"], index=2)

def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period)
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(10).std()
    return data.dropna()

def train_model(data):
    X = data[['Prev_Close','MA10','MA20','MA50','Volume','Return','Volatility']]
    y = data['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, scaler, X_scaled, r2

def predict_next_day(data):
    model, scaler, X_scaled, r2 = train_model(data)
    prev_date = data.index[-1].strftime("%Y-%m-%d")
    prev_close = data['Close'].iloc[-1].item()
    next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = model.predict([X_scaled[-1]])[0].item()
    change = ((next_day - prev_close) / prev_close) * 100
    return prev_date, prev_close, next_date, next_day, change, r2

def forecast_horizon(data, days):
    model, scaler, X_scaled, _ = train_model(data)
    prev_close = data['Close'].iloc[-1].item()
    forecasts = []
    last_features = X_scaled[-1]
    for d in range(1, days+1):
        pred = model.predict([last_features])[0].item()
        next_date = (data.index[-1] + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        change = ((pred - prev_close) / prev_close) * 100
        forecasts.append([next_date, pred, change])
        prev_close = pred
    return forecasts

# Single ticker prediction
if run_prediction:
    data = get_stock_data(ticker, period)
    prev_date, prev_close, next_date, next_day, change, r2 = predict_next_day(data)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h2>Previous Close</h2><p>{prev_date}</p><h2>${prev_close:.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h2>Predicted Next Close</h2><p>{next_date}</p><h2 class='{ 'positive' if next_day>prev_close else 'negative' }'>${next_day:.2f}</h2><p>{change:.2f}%</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h2>Model Accuracy (R²)</h2><p>Performance Metric</p><h2>{r2:.3f}</h2></div>", unsafe_allow_html=True)

# Forecast horizon
if run_forecast:
    data = get_stock_data(ticker, period)
    forecasts = forecast_horizon(data, forecast_days)
    st.subheader(f"📊 {ticker} Forecast for {forecast_days} Days Ahead")
    forecast_df = pd.DataFrame(forecasts, columns=["Date","Predicted Close","Change%"])
    st.dataframe(forecast_df)
    fig = px.line(forecast_df, x="Date", y="Predicted Close", markers=True, title=f"{ticker} Forecast Trend")
    st.plotly_chart(fig, use_container_width=True)
