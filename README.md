# 📈 Stock Price Prediction Dashboard

An interactive Streamlit web app that predicts stock prices using machine learning (Linear Regression) and visualizes results with cards, tables, and charts.
It fetches live financial data from Yahoo Finance, computes technical indicators, and predicts the next day’s closing price for selected tickers.




# 🚀 Features

 **Elegant UI**: custom background, translucent cards, styled tables, green/red highlights for gains/losses.


# Single Ticker Prediction:

Previous Close

Predicted Next Close (colored green/red)

Model Accuracy (R² score)

Download prediction as CSV


# Multi‑Ticker Comparison:

Compare multiple stocks side by side

Display results in cards, tables, and charts

Centered, styled table with colored values

Download comparison results as CSV


# Interactive Charts:
line chart showing predicted closes with accuracy labels.




# 🧠 Machine Learning

**Model**:
Linear Regression (from scikit‑learn)

**Features used**:

Previous Close

Moving Averages (10, 20, 50, 100 days)

Volume

Return

Volatility

**Target**: Closing Price

**Performance**: R² score displayed for each prediction



# 🛠 Tech Stack

**Python**

**Streamlit (UI framework)**

**yfinance (data fetching)**

**pandas / numpy (data processing)**

**scikit‑learn (machine learning)**

**plotly.express (charts)**



# 📂 Project Structure

app.py → main Streamlit application

requirements.txt → dependencies

background.jpg → background image for styling

Other files → helper scripts, notebooks, or assets

# ⚙️ How to Run

**Clone the repo**:

**Bash**

git clone https://github.com/VikramSinghShah/stock-price-prediction.git

cd stock-price-prediction


**Install dependencies**:

**Bash**

pip install -r requirements.txt


**Run the app**:

**Bash**

streamlit run app.py


Open the local URL (usually http://localhost:8501) in your browser.



# 🎯 Use Case

This project is a demonstration of machine learning applied to finance.
It’s not intended for real trading decisions, but it shows:

How to fetch and preprocess stock data

How to build a simple regression model

How to deploy ML models in a user‑friendly dashboard

# 📸 Screenshots

  **Single ticker prediction cards**
<img width="955" height="471" alt="image" src="https://github.com/user-attachments/assets/6f7cb75c-7e31-4df8-861c-c76e064b3599" />



**Multi‑ticker comparison table**
<img width="958" height="477" alt="image" src="https://github.com/user-attachments/assets/c2c6ab09-8ba7-4986-abbe-e5af2cc0d92c" />


**Interactive charts**
<img width="960" height="478" alt="image" src="https://github.com/user-attachments/assets/bf649ab9-d9a5-4fd0-bef6-470b853b81e9" />

# 🔮 Future Improvements

Add more ML models (LSTM, etc.)

Include more technical indicators

Add Excel export option with formatting
