import streamlit as st
import pandas as pd
import io
import requests
import plotly.graph_objects as go

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

# --- STREAMLIT UI ---
st.set_page_config(page_title="VWAP Live Tracker", layout="wide")
st.title("🔧 VWAP + Premarket Range Live Tracker")

symbol = st.text_input("Enter Symbol (e.g., NVDA, PLTR):", "NVDA").upper()

if symbol:
    df = fetch_intraday_data(symbol)
    if not df.empty:
        df = calculate_vwap(df)
        pre_high, pre_low = get_premarket_levels(df)

        st.subheader(f"Live VWAP Chart: {symbol}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], mode='lines', name='VWAP'))

        if pre_high and pre_low:
            fig.add_hline(y=pre_high, line=dict(color="green", dash="dot"), name="Premarket High")
            fig.add_hline(y=pre_low, line=dict(color="red", dash="dot"), name="Premarket Low")

        fig.update_layout(height=600, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Premarket High:** {pre_high:.2f} | **Low:** {pre_low:.2f}")
