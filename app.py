import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Background styling
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
    .positive {color: green; font-weight: bold; font-size: 32px;}
    .negative {color: red; font-weight: bold; font-size: 32px;}
    .section-title {text-align: center; font-size: 30px; font-weight: bold; margin-top: 30px; margin-bottom: 20px; color:#FFFFFF;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📈 Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
model_choice = st.sidebar.radio("Choose Model", ["Linear Regression","Random Forest"])
run_prediction = st.sidebar.button("Run Prediction")

multi_tickers = st.sidebar.multiselect("Compare Multiple Tickers", ["AAPL","TSLA","MSFT"], default=["AAPL","TSLA","MSFT"])
run_comparison = st.sidebar.button("Run Comparison")
display_mode = st.sidebar.radio("Show in Comparison", ["Table Only","Charts Only","Table + Charts"], index=2)

def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period)
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA100'] = data['Close'].rolling(100).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(10).std()
    return data.dropna()

def predict_stock(data, model_choice):
    X = data[['Prev_Close','MA10','MA20','MA50','MA100','Volume','Return','Volatility']]
    y = data['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)

    prev_date = data.index[-1].strftime("%Y-%m-%d")
    prev_close = data['Close'].iloc[-1].item()
    next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = model.predict([X_scaled[-1]])[0].item()
    change = ((next_day - prev_close) / prev_close) * 100

    return prev_date, prev_close, next_date, next_day, change, r2

# Single ticker prediction
if run_prediction:
    data = get_stock_data(ticker, period)
    prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data, model_choice)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h2>Previous Close</h2><p>{prev_date}</p><h2>${prev_close:.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h2>Predicted Next Close</h2><p>{next_date}</p><h2 class='{ 'positive' if next_day>prev_close else 'negative' }'>${next_day:.2f}</h2><p>{change:.2f}%</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h2>Model Accuracy (R²)</h2><p>{model_choice}</p><h2>{r2:.3f}</h2></div>", unsafe_allow_html=True)

# Multi-ticker comparison
if run_comparison:
    results = []
    for t in multi_tickers:
        data = get_stock_data(t, period)
        prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data, model_choice)
        results.append([t, prev_date, prev_close, next_date, next_day, change, r2])

    comp_df = pd.DataFrame(results, columns=["Ticker","Prev Date","Prev Close","Next Date","Predicted Close","Change%","R²"])
    st.markdown("<div class='section-title'>📊 Multi‑Ticker Comparison</div>", unsafe_allow_html=True)

    if display_mode in ["Table Only","Table + Charts"]:
        st.dataframe(comp_df.style.apply(
            lambda row: ['color: green; font-weight:bold;' if row["Predicted Close"] > row["Prev Close"] and col=="Predicted Close"
                         else 'color: red; font-weight:bold;' if row["Predicted Close"] < row["Prev Close"] and col=="Predicted Close"
                         else '' for col in comp_df.columns], axis=1
        ), use_container_width=True)

    if display_mode in ["Charts Only","Table + Charts"]:
        with st.expander("📊 See Interactive Charts"):
            df_melt = comp_df.melt(id_vars="Ticker", value_vars=["Prev Close","Predicted Close"], var_name="Type", value_name="Price")
            fig1 = px.bar(df_melt, x="Ticker", y="Price", color="Type", barmode="group",
                          title="Predicted vs Previous Close")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.bar(comp_df, x="Ticker", y="Change%", color="Change%",
                          title="Predicted Percentage Change",
                          color_continuous_scale=["red","green"])
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<p style='text-align:center; font-size:16px;'>🟢 Profit | 🔴 Loss | R² = Model Accuracy</p>", unsafe_allow_html=True)
