import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

FINNHUB_API_KEY = "cutllthr01qv6ijj0i9gcutllthr01qv6ijj0ia0"

popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META"]

def fetch_analyst_ratings(stock_ticker):
    """Fetch analyst recommendations and price targets for a stock using Finnhub API."""
    if not stock_ticker:
        st.warning("⚠️ Please select a stock ticker.")
        return
    
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={stock_ticker}&token={FINNHUB_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"❌ Failed to fetch analyst ratings. Check API key or stock ticker: {stock_ticker}")
        return
    
    data = response.json()
    if not data:
        st.warning(f"⚠️ No analyst ratings available for {stock_ticker}.")
        return

    latest_rec = pd.DataFrame(data).iloc[0]  #
    
    st.subheader(f"📊 Analyst Ratings & Price Targets for {stock_ticker}")
    total_analysts = latest_rec['strongBuy'] + latest_rec['buy'] + latest_rec['hold'] + latest_rec['sell'] + latest_rec['strongSell']
    st.write(f"**Total Analysts Covering:** {total_analysts}")

    
    sentiment_data = {
        "Strong Buy": latest_rec['strongBuy'],
        "Buy": latest_rec['buy'],
        "Hold": latest_rec['hold'],
        "Sell": latest_rec['sell'],
        "Strong Sell": latest_rec['strongSell'],
    }

    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sentiment_data.keys()),
        y=list(sentiment_data.values()),
        marker_color=['green', 'lightgreen', 'gray', 'orange', 'red']
    ))
    fig.update_layout(title="Analyst Recommendations", xaxis_title="Rating", yaxis_title="Count")
    st.plotly_chart(fig)

    
    sentiment_score = (latest_rec['strongBuy'] * 2 + latest_rec['buy'] * 1 - latest_rec['sell'] * 1 - latest_rec['strongSell'] * 2) / total_analysts
    sentiment = "🔵 Neutral"
    if sentiment_score > 0.5:
        sentiment = "🟢 Bullish"
    elif sentiment_score < -0.5:
        sentiment = "🔴 Bearish"

    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={"text": "Market Sentiment"},
        gauge={
            "axis": {"range": [-1, 1]},
            "steps": [
                {"range": [-1, -0.5], "color": "red"},
                {"range": [-0.5, 0.5], "color": "gray"},
                {"range": [0.5, 1], "color": "green"}
            ],
            "bar": {"color": "blue"}
        }
    ))
    st.plotly_chart(fig_gauge)
    
    st.subheader(f"📈 Overall Analyst Sentiment: **{sentiment}**")


def show_analyst_ratings():
    st.title("📈 Analyst Ratings & Price Targets")

    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_ticker = st.selectbox("🔍 Select Stock Ticker:", options=popular_stocks, index=0)
    with col2:
        stock_ticker_custom = st.text_input("Or Enter Custom Ticker:").upper()
        if stock_ticker_custom:
            stock_ticker = stock_ticker_custom  

    if stock_ticker:
        fetch_analyst_ratings(stock_ticker)
