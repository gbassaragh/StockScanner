import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import io

# --- CONFIG ---
API_KEY = "YR2MYD1XJHOX1UB7"
INTERVAL = "1min"
AV_URL = "https://www.alphavantage.co/query"

# --- FUNCTIONS ---
def fetch_intraday_data(symbol):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": INTERVAL,
        "outputsize": "full",
        "apikey": API_KEY,
        "datatype": "csv"
    }
    response = requests.get(AV_URL, params=params)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        st.error("Error fetching data from Alpha Vantage")
        return pd.DataFrame()

def calculate_vwap(df):
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_x_vol"] = df["typical_price"] * df["volume"]
    df["cum_tp_x_vol"] = df["tp_x_vol"].cumsum()
    df["cum_vol"] = df["volume"].cumsum()
    df["vwap"] = df["cum_tp_x_vol"] / df["cum_vol"]
    return df

def get_premarket_levels(df):
    premarket = df[df["timestamp"].dt.time < pd.to_datetime("09:30").time()]
    if premarket.empty:
        return None, None
    return premarket["high"].max(), premarket["low"].min()

def detect_vwap_cross(df):
    df["vwap_cross"] = ""
    for i in range(1, len(df)):
        if df.loc[i-1, "close"] < df.loc[i-1, "vwap"] and df.loc[i, "close"] > df.loc[i, "vwap"]:
            df.loc[i, "vwap_cross"] = "↑ Cross"
        elif df.loc[i-1, "close"] > df.loc[i-1, "vwap"] and df.loc[i, "close"] < df.loc[i, "vwap"]:
            df.loc[i, "vwap_cross"] = "↓ Cross"
    return df

def detect_support_resistance(df):
    df["support"] = df["low"].rolling(window=20, min_periods=1).min()
    df["resistance"] = df["high"].rolling(window=20, min_periods=1).max()
    return df

# --- STREAMLIT UI ---
st.set_page_config(page_title="VWAP Live Tracker", layout="wide")
st.title("🔧 VWAP + Premarket Range Live Tracker")

symbol = st.text_input("Enter Symbol (e.g., NVDA, PLTR):", "NVDA").upper()

if symbol:
    df = fetch_intraday_data(symbol)
    if not df.empty:
        df = calculate_vwap(df)
        df = detect_vwap_cross(df)
        df = detect_support_resistance(df)
        pre_high, pre_low = get_premarket_levels(df)

        st.subheader(f"Live VWAP Chart: {symbol}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], mode='lines', name='VWAP'))

        if pre_high and pre_low:
            fig.add_hline(y=pre_high, line=dict(color="green", dash="dot"), name="Premarket High")
            fig.add_hline(y=pre_low, line=dict(color="red", dash="dot"), name="Premarket Low")

        # Plot VWAP Cross tags
        cross_df = df[df["vwap_cross"] != ""]
        fig.add_trace(go.Scatter(
            x=cross_df["timestamp"],
            y=cross_df["close"],
            mode="markers+text",
            text=cross_df["vwap_cross"],
            textposition="top center",
            marker=dict(size=8, color="orange"),
            name="VWAP Cross"
        ))

        # Optional: Show support/resistance zones (last levels)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["support"], line=dict(dash="dot", color="lightblue"), name="Support"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["resistance"], line=dict(dash="dot", color="lightcoral"), name="Resistance"))

        fig.update_layout(height=600, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Premarket High:** {pre_high:.2f} | **Low:** {pre_low:.2f}")
