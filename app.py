import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# ✅ Modern classy styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/VikramSinghShah/stock-price-prediction/main/background.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0,0,0,0.55);
        z-index: -1;
    }
    .main-title {
        font-size: 48px; font-weight: bold; color: #FDFEFE;
        text-align: center; margin-bottom: 30px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    .card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        padding: 25px; border-radius: 15px;
        text-align: center; margin: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s ease-in-out;
    }
    .card:hover { transform: scale(1.03); }
    .card h2 {margin: 0; font-size: 30px; color: #FDFEFE; font-weight:bold;}
    .card p {margin: 8px 0; font-size: 18px; color: #D5DBDB;}
    .positive {color: #2ECC71; font-weight: bold; font-size: 28px;}
    .negative {color: #E74C3C; font-weight: bold; font-size: 28px;}
    .section-title {
        text-align: center; font-size: 32px; font-weight: bold;
        margin-top: 40px; margin-bottom: 20px; color:#FDFEFE;
        text-shadow: 1px 1px 6px rgba(0,0,0,0.6);
    }
    .dataframe {
        background: rgba(255,255,255,0.08);
        color: #FDFEFE;
        border-radius: 10px;
        padding: 10px;
        margin: auto;
        width: 90%;
    }
    th, td {
        text-align: center;
        color: #FDFEFE;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title"> Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# ✅ Single ticker input (dynamic)
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()

period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("Run Prediction")

# ✅ Multi-ticker input (popular + custom)
popular_tickers = ["AAPL","TSLA","MSFT","AMZN","GOOG","META","NFLX","NVDA"]

selected_popular = st.sidebar.multiselect(
    "Choose Popular Tickers",
    popular_tickers,
    default=["AAPL","TSLA","MSFT"]
)

custom_input = st.sidebar.text_input("Add Custom Tickers (comma separated)", "")

# Merge both popular and custom tickers
multi_tickers = selected_popular + [t.strip().upper() for t in custom_input.split(",") if t.strip()]

run_comparison = st.sidebar.button("Run Comparison")

display_mode = st.sidebar.radio(
    "Show in Comparison",
    ["Cards","Table","Charts","Cards + Table + Charts"],
    index=0
)

# ✅ Functions
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

def predict_stock(data):
    X = data[['Prev_Close','MA10','MA20','MA50','MA100','Volume','Return','Volatility']]
    y = data['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    prev_date = data.index[-1].strftime("%Y-%m-%d")
    prev_close = data['Close'].iloc[-1].item()
    next_date = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = model.predict([X_scaled[-1]])[0].item()
    change = ((next_day - prev_close) / prev_close) * 100
    return prev_date, prev_close, next_date, next_day, change, r2

# ✅ Single ticker prediction
if run_prediction and ticker:
    try:
        data = get_stock_data(ticker, period)
        prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)
        r2_percent = r2 * 100
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'><h2>Previous Close</h2><p>{prev_date}</p><h2>${prev_close:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><h2>Predicted Next Close</h2><p>{next_date}</p><h2 class='{ 'positive' if next_day>prev_close else 'negative' }'>${next_day:.2f}</h2><p>{change:.2f}%</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><h2>Model Accuracy (R²)</h2><p>Linear Regression</p><h2>{r2_percent:.1f}%</h2></div>", unsafe_allow_html=True)
        export_df = pd.DataFrame({"Date":[prev_date,next_date],"Close":[prev_close,next_day]})
        st.download_button("⬇️ Download Prediction Data", export_df.to_csv(index=False), "prediction.csv", "text/csv")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

# ✅ Multi-ticker comparison
if run_comparison and multi_tickers:
    results = []
    for t in multi_tickers:
        try:
            data = get_stock_data(t, period)
            prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)
            r2_percent = r2 * 100
            results.append({
                "Ticker": t,
                "Prev Date": prev_date,
                "Prev Close": f"${prev_close:.2f}",
                "Next Date": next_date,
                "Predicted Close": f"${next_day:.2f}",
                "Change%": f"{change:.2f}%",
                "Accuracy (R²)": f"{r2_percent:.1f}%"
            })
        except Exception as e:
            st.warning(f"Could not fetch data for {t}: {e}")

    if results:
        chart_df = pd.DataFrame(results)

        # ✅ Cards
        if display_mode in ["Cards","Cards + Table + Charts"]:
            st.markdown("<div class='section-title'>📋 Multi‑Ticker Cards</div>", unsafe_allow_html=True)
            for i in range(0, len(chart_df), 3):
                row = chart_df.iloc[i:i+3]
                cols = st.columns(len(row))
                for j, (_, r) in enumerate(row.iterrows()):
                    change_value = float(r["Change%"].replace("%",""))
                    change_class = "positive" if change_value >= 0 else "negative"
                    cols[j].markdown(
                        f"""
                        <div class='card'>
                            <h2>{r['Ticker']}</h2>
                            <p><b>Prev Date:</b> {r['Prev Date']}</p>
                            <h2 style="font-size:26px;">{r['Prev Close']}</h2>
                            <p><b>Next Date:</b> {r['Next Date']}</p>
                            <h2 style="font-size:26px;" class="{change_class}">{r['Predicted Close']}</h2>
                            <p><b>Change:</b> <span class="{change_class}">{r['Change%']}</span></p>
                            <p><b>Accuracy:</b> {r['Accuracy (R²)']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True)