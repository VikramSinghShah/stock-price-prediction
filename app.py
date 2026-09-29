import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np   # ✅ Needed for .item()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Stock Price Prediction (Linear Regression)")

def get_stock_data(ticker):
    data = yf.download(ticker, period="2y")
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    data = data.dropna()
    return data

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")

if st.button("Predict"):
    data = get_stock_data(ticker)

    X = data[['Prev_Close','MA10','MA20','Volume']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("R² Score:", model.score(X_test, y_test))

    results = pd.DataFrame({
        "Actual": y_test.to_numpy().ravel(),
        "Predicted": y_pred.ravel()
    })

    # ✅ Predict next day
    next_day = model.predict([X.iloc[-1].values])[0]

    # ✅ Safely convert to Python float
    if isinstance(next_day, (np.ndarray, list)):
        next_day = next_day.item()
    else:
        next_day = float(next_day)

    # ✅ Correct indentation: st.success must be outside the else block
    st.success(f"Predicted Next Day Closing Price for {ticker}: ${next_day:.2f}")
