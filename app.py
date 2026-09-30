import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
        background-color: rgba(0,0,0,0.5); /* dark overlay */
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

st.markdown('<div class="main-title">📈 Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("Run Prediction")

multi_tickers = st.sidebar.multiselect("Compare Multiple Tickers", ["AAPL","TSLA","MSFT"], default=["AAPL","TSLA","MSFT"])
run_comparison = st.sidebar.button("Run Comparison")
display_mode = st.sidebar.radio("Show in Comparison", ["Table Only","Charts Only","Table + Charts"], index=2)

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

# Single ticker prediction
if run_prediction:
    data = get_stock_data(ticker, period)
    if data is None or data.empty:
        st.error("❌ Invalid ticker or no data available.")
    else:
        prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card' title='Previous day closing price'><h2>Previous Close</h2><p>{prev_date}</p><h2>${prev_close:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card' title='Green = profit, Red = loss'><h2>Predicted Next Close</h2><p>{next_date}</p><h2 class='{ 'positive' if next_day>prev_close else 'negative' }'>${next_day:.2f}</h2><p>{change:.2f}%</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card' title='R² shows how well the model fits'><h2>Model Accuracy (R²)</h2><p>Performance Metric</p><h2>{r2:.3f}</h2></div>", unsafe_allow_html=True)

        export_df = pd.DataFrame({"Date":[prev_date,next_date],"Close":[prev_close,next_day],"Change%":[0,change]})
        st.download_button("📥 Download Predictions (CSV)", export_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# Multi-ticker comparison
if run_comparison:
    results = []
    for t in multi_tickers:
        data = get_stock_data(t, period)
        if data is not None and not data.empty:
            prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)
            results.append([t, prev_date, prev_close, next_date, next_day, change, r2])

    if results:
        comp_df = pd.DataFrame(results, columns=["Ticker","Prev Date","Prev Close","Next Date","Predicted Close","Change%","R²"])
        st.markdown("<div class='section-title'>📊 Multi‑Ticker Comparison</div>", unsafe_allow_html=True)

        def highlight_pred(row):
            return ['color: green; font-weight:bold;' if row["Predicted Close"] > row["Prev Close"] and col=="Predicted Close"
                    else 'color: red; font-weight:bold;' if row["Predicted Close"] < row["Prev Close"] and col=="Predicted Close"
                    else '' for col in comp_df.columns]
        styled_df = comp_df.style.apply(highlight_pred, axis=1)

        if display_mode in ["Table Only","Table + Charts"]:
            st.dataframe(styled_df, use_container_width=True)

        if display_mode in ["Charts Only","Table + Charts"]:
            with st.expander("📊 See Interactive Charts"):
                df_melt = comp_df.melt(id_vars="Ticker", value_vars=["Prev Close","Predicted Close"], var_name="Type", value_name="Price")
                fig1 = px.bar(df_melt, x="Ticker", y="Price", color="Type", barmode="group",
                              title="Predicted vs Previous Close",
                              labels={"Price":"Price","Ticker":"Stock"},
                              hover_data={"Price":True,"Type":True})
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.bar(comp_df, x="Ticker", y="Change%", color="Change%",
                              title="Predicted Percentage Change",
                              labels={"Change%":"% Change","Ticker":"Stock"},
                              color_continuous_scale=["red","green"],
                              hover_data={"Change%":True,"R²":True})
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("<p style='text-align:center; font-size:16px;'>🟢 Profit | 🔴 Loss | R² = Model Accuracy</p>", unsafe_allow_html=True)

        st.download_button("📥 Download Comparison (CSV)", comp_df.to_csv(index=False).encode("utf-8"), "comparison.csv", "text/csv")
