import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN APP BACKGROUND
       ========================= */

    .stApp {
        background-image:
            url("https://raw.githubusercontent.com/VikramSinghShah/stock-price-prediction/main/background.jpg");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        font-family: 'Segoe UI', sans-serif;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;

        background-color: rgba(0, 0, 0, 0.55);

        z-index: -1;
    }


    /* =========================
       MAIN TITLE
       ========================= */

    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #FDFEFE;
        text-align: center;
        margin-bottom: 30px;

        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }


    /* =========================
       GLASS CARDS
       ========================= */

    .card {
        background: rgba(255,255,255,0.08);

        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);

        padding: 25px;
        border-radius: 15px;

        text-align: center;
        margin: 10px;

        border: 1px solid rgba(255,255,255,0.12);

        box-shadow:
            0 4px 20px rgba(0,0,0,0.4);

        transition: transform 0.2s ease-in-out;
    }

    .card:hover {
        transform: scale(1.03);
    }

    .card h2 {
        margin: 0;
        font-size: 30px;
        color: #FDFEFE;
        font-weight: bold;
    }

    .card p {
        margin: 8px 0;
        font-size: 18px;
        color: #D5DBDB;
    }


    /* =========================
       POSITIVE / NEGATIVE
       ========================= */

    .positive {
        color: #2ECC71;
        font-weight: bold;
        font-size: 28px;
    }

    .negative {
        color: #E74C3C;
        font-weight: bold;
        font-size: 28px;
    }


    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        text-align: center;

        font-size: 32px;
        font-weight: bold;

        margin-top: 40px;
        margin-bottom: 20px;

        color: #FDFEFE;

        text-shadow:
            1px 1px 6px rgba(0,0,0,0.6);
    }


    /* =========================
       TABLE CONTAINER
       ========================= */

    .table-container {
        background: rgba(255,255,255,0.08);

        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);

        border-radius: 15px;

        padding: 15px;

        border: 1px solid rgba(255,255,255,0.12);

        box-shadow:
            0 4px 20px rgba(0,0,0,0.4);

        margin-top: 10px;
        margin-bottom: 25px;
    }


    /* =========================
       STREAMLIT DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {
        background: transparent !important;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] iframe {
        background: transparent !important;
    }


    /* =========================
       CHART GLASS CONTAINER
       ========================= */

    .chart-container {
        background: rgba(255,255,255,0.08);

        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);

        border-radius: 15px;

        padding: 15px;

        border: 1px solid rgba(255,255,255,0.12);

        box-shadow:
            0 4px 20px rgba(0,0,0,0.4);

        margin-top: 10px;
        margin-bottom: 25px;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: rgba(10,10,10,0.75);

        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border-radius: 10px;

        border: 1px solid rgba(255,255,255,0.2);

        background: rgba(255,255,255,0.10);

        color: white;

        font-weight: bold;

        transition: 0.2s;
    }

    .stButton > button:hover {
        background: rgba(255,255,255,0.20);

        border-color: rgba(255,255,255,0.35);

        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📈 Elegant Stock Price Prediction Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    "AAPL"
).upper().strip()

period = st.sidebar.selectbox(
    "Data Period",
    ["1y", "2y", "5y"],
    index=1
)

run_prediction = st.sidebar.button(
    "Run Prediction"
)


multi_tickers = st.sidebar.multiselect(
    "Compare Multiple Tickers",
    ["AAPL", "TSLA", "MSFT"],
    default=["AAPL", "TSLA", "MSFT"]
)

run_comparison = st.sidebar.button(
    "Run Comparison"
)


display_mode = st.sidebar.radio(
    "Show in Comparison",
    [
        "Cards",
        "Table",
        "Charts",
        "Cards + Table + Charts"
    ],
    index=0
)


# ============================================================
# GET STOCK DATA
# ============================================================

def get_stock_data(ticker, period):

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No stock data found for ticker: {ticker}"
        )

    # Handle newer yfinance MultiIndex format
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.get_level_values(0)
        except Exception:
            pass

    required_columns = [
        "Close",
        "Volume"
    ]

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(
                f"Required column '{col}' was not found."
            )

    # Make sure Close is a Series
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    data["Close"] = pd.to_numeric(
        close,
        errors="coerce"
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    )

    # Technical indicators
    data["Prev_Close"] = data["Close"].shift(1)

    data["MA10"] = (
        data["Close"]
        .rolling(10)
        .mean()
    )

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["MA100"] = (
        data["Close"]
        .rolling(100)
        .mean()
    )

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Volatility"] = (
        data["Close"]
        .rolling(10)
        .std()
    )

    data = data.dropna()

    if len(data) < 20:
        raise ValueError(
            "Not enough historical data available "
            "to build the prediction model."
        )

    return data


# ============================================================
# PREDICTION MODEL
# ============================================================

def predict_stock(data):

    features = [
        "Prev_Close",
        "MA10",
        "MA20",
        "MA50",
        "MA100",
        "Volume",
        "Return",
        "Volatility"
    ]

    X = data[features]
    y = data["Close"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    r2 = model.score(
        X_test,
        y_test
    )

    prev_date = data.index[-1].strftime(
        "%Y-%m-%d"
    )

    prev_close = float(
        data["Close"].iloc[-1]
    )

    next_date = (
        data.index[-1]
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    next_day = float(
        model.predict(
            X_scaled[-1].reshape(1, -1)
        )[0]
    )

    change = (
        (next_day - prev_close)
        / prev_close
    ) * 100

    return (
        prev_date,
        prev_close,
        next_date,
        next_day,
        change,
        r2
    )


# ============================================================
# SINGLE TICKER PREDICTION
# ============================================================

if run_prediction:

    try:

        data = get_stock_data(
            ticker,
            period
        )

        (
            prev_date,
            prev_close,
            next_date,
            next_day,
            change,
            r2
        ) = predict_stock(data)

        r2_percent = r2 * 100


        # -------------------------
        # CARDS
        # -------------------------

        col1, col2, col3 = st.columns(3)


        # Previous Close
        col1.markdown(
            f"""
            <div class="card">

                <h2>Previous Close</h2>

                <p>{prev_date}</p>

                <h2>${prev_close:.2f}</h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Predicted Close
        color_class = (
            "positive"
            if next_day > prev_close
            else "negative"
        )

        col2.markdown(
            f"""
            <div class="card">

                <h2>Predicted Next Close</h2>

                <p>{next_date}</p>

                <h2 class="{color_class}">
                    ${next_day:.2f}
                </h2>

                <p>{change:.2f}%</p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # R2
        col3.markdown(
            f"""
            <div class="card">

                <h2>Model Accuracy (R²)</h2>

                <p>Linear Regression</p>

                <h2>{r2_percent:.1f}%</h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------
        # DOWNLOAD CSV
        # -------------------------

        export_df = pd.DataFrame(
            {
                "Date": [
                    prev_date,
                    next_date
                ],

                "Close": [
                    prev_close,
                    next_day
                ],

                "Change%": [
                    0,
                    change
                ],

                "R²%": [
                    r2_percent,
                    r2_percent
                ]
            }
        )

        st.download_button(
            "📥 Download Predictions (CSV)",
            export_df.to_csv(
                index=False
            ).encode("utf-8"),

            "predictions.csv",

            "text/csv"
        )


    except Exception as e:

        st.error(
            f"❌ Unable to generate prediction: {e}"
        )


# ============================================================
# MULTI-TICKER COMPARISON
# ============================================================

if run_comparison:

    if not multi_tickers:

        st.warning(
            "Please select at least one ticker."
        )

    else:

        results = []

        errors = []


        for t in multi_tickers:

            try:

                data = get_stock_data(
                    t,
                    period
                )

                (
                    prev_date,
                    prev_close,
                    next_date,
                    next_day,
                    change,
                    r2
                ) = predict_stock(data)

                results.append(
                    [
                        t,
                        prev_date,
                        prev_close,
                        next_date,
                        next_day,
                        change,
                        r2
                    ]
                )

            except Exception as e:

                errors.append(
                    f"{t}: {e}"
                )


        # -------------------------
        # SHOW ERRORS
        # -------------------------

        if errors:

            for error in errors:

                st.warning(
                    f"⚠️ {error}"
                )


        if results:

            comp_df = pd.DataFrame(
                results,

                columns=[
                    "Ticker",
                    "Prev Date",
                    "Prev Close",
                    "Next Date",
                    "Predicted Close",
                    "Change%",
                    "R²"
                ]
            )


            # Convert R2 into percentage
            comp_df["R²%"] = (
                comp_df["R²"] * 100
            )

            # Remove original R2 column
            comp_df = comp_df.drop(
                columns=["R²"]
            )


            # ==================================================
            # SECTION TITLE
            # ==================================================

            st.markdown(
                '<div class="section-title">'
                '📊 Multi-Ticker Comparison'
                '</div>',
                unsafe_allow_html=True
            )


            # ==================================================
            # CARD VIEW
            # ==================================================

            if display_mode in [
                "Cards",
                "Cards + Table + Charts"
            ]:

                cols = st.columns(
                    len(comp_df)
                )

                for idx, row in comp_df.iterrows():

                    color_class = (
                        "positive"
                        if row["Predicted Close"]
                        > row["Prev Close"]
                        else "negative"
                    )

                    cols[idx].markdown(
                        f"""
                        <div class="card">

                            <h2>
                                {row["Ticker"]}
                            </h2>

                            <p>
                                <b>Prev Date:</b>
                                {row["Prev Date"]}
                            </p>

                            <p>
                                <b>Prev Close:</b>
                                ${row["Prev Close"]:.2f}
                            </p>

                            <p>
                                <b>Next Date:</b>
                                {row["Next Date"]}
                            </p>

                            <h2 class="{color_class}">
                                ${row["Predicted Close"]:.2f}
                            </h2>

                            <p>
                                <b>Change:</b>
                                {row["Change%"]:.2f}%
                            </p>

                            <p>
                                <b>R²:</b>
                                {row["R²%"]:.1f}%
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # ==================================================
            # TABLE VIEW
            # ==================================================

            if display_mode in [
                "Table",
                "Cards + Table + Charts"
            ]:

                st.markdown(
                    """
                    <div class="section-title">
                        📋 Comparison Table
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # Create a display copy
                table_df = comp_df.copy()


                # Format numeric columns
                table_df["Prev Close"] = (
                    table_df["Prev Close"]
                    .map(lambda x: f"${x:.2f}")
                )

                table_df["Predicted Close"] = (
                    table_df["Predicted Close"]
                    .map(lambda x: f"${x:.2f}")
                )

                table_df["Change%"] = (
                    table_df["Change%"]
                    .map(lambda x: f"{x:.2f}%")
                )

                table_df["R²%"] = (
                    table_df["R²%"]
                    .map(lambda x: f"{x:.1f}%")
                )


                # Styled HTML table
                html_table = """
                <div class="table-container">

                    <table style="
                        width:100%;
                        border-collapse:collapse;
                        color:#FDFEFE;
                        font-size:16px;
                        background:rgba(255,255,255,0.03);
                    ">

                        <thead>

                            <tr style="
                                background:rgba(255,255,255,0.12);
                                border-bottom:
                                1px solid rgba(255,255,255,0.2);
                            ">
                """


                # Header
                for column in table_df.columns:

                    html_table += f"""
                        <th style="
                            padding:14px;
                            text-align:center;
                            color:#FFFFFF;
                            font-weight:bold;
                        ">
                            {column}
                        </th>
                    """


                html_table += """
                            </tr>
                        </thead>

                        <tbody>
                """


                # Rows
                for _, row in table_df.iterrows():

                    html_table += """
                        <tr style="
                            border-bottom:
                            1px solid
                            rgba(255,255,255,0.10);
                        ">
                    """


                    for column in table_df.columns:

                        value = row[column]

                        # Highlight predicted close
                        if column == "Predicted Close":

                            predicted = float(
                                comp_df.loc[
                                    comp_df["Ticker"]
                                    == row["Ticker"],
                                    "Predicted Close"
                                ].iloc[0]
                            )

                            previous = float(
                                comp_df.loc[
                                    comp_df["Ticker"]
                                    == row["Ticker"],
                                    "Prev Close"
                                ].iloc[0]
                            )

                            if predicted > previous:

                                text_color = "#2ECC71"

                            else:

                                text_color = "#E74C3C"

                        else:

                            text_color = "#FDFEFE"


                        html_table += f"""
                            <td style="
                                padding:14px;
                                text-align:center;
                                color:{text_color};
                            ">
                                {value}
                            </td>
                        """


                    html_table += """
                        </tr>
                    """


                html_table += """
                        </tbody>

                    </table>

                </div>
                """


                st.markdown(
                    html_table,
                    unsafe_allow_html=True
                )


            # ==================================================
            # CHART VIEW
            # ==================================================

            if display_mode in [
                "Charts",
                "Cards + Table + Charts"
            ]:

                st.markdown(
                    """
                    <div class="section-title">
                        📈 Price Comparison Chart
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                chart_data = comp_df[
                    [
                        "Ticker",
                        "Prev Close",
                        "Predicted Close"
                    ]
                ].copy()


                # Create Plotly chart
                fig = px.bar(
                    chart_data,

                    x="Ticker",

                    y=[
                        "Prev Close",
                        "Predicted Close"
                    ],

                    barmode="group",

                    title="Previous Close vs Predicted Close",

                    labels={
                        "value": "Price ($)",
                        "variable": "Price Type",
                        "Ticker": "Stock"
                    },

                    color_discrete_sequence=[
                        "#5DADE2",
                        "#2ECC71"
                    ]
                )


                # Same glass/dark background
                fig.update_layout(

                    paper_bgcolor="rgba(255,255,255,0.08)",

                    plot_bgcolor="rgba(255,255,255,0.03)",

                    font=dict(
                        color="#FDFEFE",
                        family="Segoe UI"
                    ),

                    title=dict(
                        font=dict(
                            size=24,
                            color="#FDFEFE"
                        )
                    ),

                    legend=dict(
                        font=dict(
                            color="#FDFEFE"
                        ),

                        bgcolor="rgba(255,255,255,0.05)"
                    ),

                    xaxis=dict(
                        title_font=dict(
                            color="#D5DBDB"
                        ),

                        tickfont=dict(
                            color="#D5DBDB"
                        ),

                        gridcolor="rgba(255,255,255,0.08)"
                    ),

                    yaxis=dict(
                        title_font=dict(
                            color="#D5DBDB"
                        ),

                        tickfont=dict(
                            color="#D5DBDB"
                        ),

                        gridcolor="rgba(255,255,255,0.08)"
                    ),

                    margin=dict(
                        l=40,
                        r=40,
                        t=70,
                        b=40
                    ),

                    hoverlabel=dict(
                        bgcolor="#1C2833",
                        font_color="white"
                    )
                )


                # Rounded-looking chart container
                st.markdown(
                    '<div class="chart-container">',
                    unsafe_allow_html=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )
