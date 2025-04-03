import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

st.title("Intraday VWAP Viewer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with timestamp, price, volume", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Parse timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # Calculate VWAP
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_pv"] = (df["price"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]

    # Create Plotly figure
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["price"],
        mode="lines", name="Price", line=dict(color="blue")
    ))

    # VWAP line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["vwap"],
        mode="lines", name="VWAP", line=dict(color="orange", dash="dot")
    ))

    # Market open/close vertical lines for each day
    trading_days = df["timestamp"].dt.date.unique()
    for day in trading_days:
        open_time = pd.Timestamp(f"{day} 09:30:00")
        close_time = pd.Timestamp(f"{day} 16:00:00")

        fig.add_vline(
            x=open_time,
            line=dict(color="green", width=1, dash="dot"),
            annotation_text=f"Open {pd.to_datetime(day).strftime('%m%d')}",
            annotation_position="top left"
        )
        fig.add_vline(
            x=close_time,
            line=dict(color="red", width=1, dash="dot"),
            annotation_text=f"Close {pd.to_datetime(day).strftime('%m%d')}",
            annotation_position="top right"
        )

    # Update layout
    fig.update_layout(
        title="Intraday Price with VWAP and Market Open/Close",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis=dict(
            tickformat="%m%d",
            showspikes=True,
            showgrid=True,
            showline=True,
            mirror=True,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(showspikes=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show raw data
    with st.expander("Show Raw Data"):
        st.dataframe(df)

