import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import datetime

# Constants
AV_API_KEY = st.secrets["AV_API_KEY"]  # Store your API key securely in Streamlit secrets
AV_URL = "https://www.alphavantage.co/query"

# Function to fetch intraday data (1-minute)
def fetch_intraday_data(symbol):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": AV_API_KEY,
        "datatype": "csv",
        "outputsize": "full"
    }
    response = requests.get(AV_URL, params=params)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

# Function to calculate VWAP
def calculate_vwap(df):
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_vol_tp"] = (df["tp"] * df["volume"]).cumsum()
    df["cum_vol"] = df["volume"].cumsum()
    df["vwap"] = df["cum_vol_tp"] / df["cum_vol"]
    return df

# Premarket high/low (assumes premarket is before 09:30)
def get_premarket_levels(df):
    premarket = df[df["timestamp"].dt.time < datetime.time(9, 30)]
    return premarket["high"].max(), premarket["low"].min()

# App
st.set_page_config(page_title="Stock VWAP Scanner", layout="wide")
st.title("📊 Stock VWAP Scanner")

symbol = st.text_input("Enter Symbol (e.g., NVDA, PLTR):", "NVDA").upper()

if symbol:
    df = fetch_intraday_data(symbol)
    if not df.empty:
        df = calculate_vwap(df)
        pre_high, pre_low = get_premarket_levels(df)

        # Time slicer
        min_ts = df["timestamp"].min().to_pydatetime()
        max_ts = df["timestamp"].max().to_pydatetime()
        start_time, end_time = st.slider(
            "Select time range:",
            min_value=min_ts,
            max_value=max_ts,
            value=(min_ts, max_ts),
            format="HH:mm"
        )
        df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], name="VWAP"))

        # Premarket lines
        fig.add_hline(y=pre_high, line_dash="dash", line_color="green", annotation_text="Premarket High")
        fig.add_hline(y=pre_low, line_dash="dash", line_color="red", annotation_text="Premarket Low")

        fig.update_layout(title=f"{symbol} - Price & VWAP", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data returned. Check symbol or API usage limit.")
