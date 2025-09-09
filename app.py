import streamlit as st
from twitter_sentiment_analysis import run_twitter_sentiment_analysis
from stock_sentiment_analysis import run_stock_sentiment_analysis
from stock_trend_prediction import run_stock_trend_prediction
from investors_info import show_investors
from analyst_ratings import show_analyst_ratings


st.set_page_config(page_title="StockVision AI - Stock Market Dashboard", layout="wide")
st.title("📈 StockVision AI - Market Intelligence Dashboard")

# Sidebar Navigation
pages = {
    "💹 Stock Sentiment Analysis": run_stock_sentiment_analysis,   
    "🐦 Twitter Sentiment Analysis": run_twitter_sentiment_analysis,
    "📊 Stock Trend Prediction": run_stock_trend_prediction,
    "🏦 Institutional & Major Investors": show_investors,
    "📑 Analyst Ratings & Price Targets": show_analyst_ratings,
}


page_selection = st.sidebar.selectbox("🔍 Select a page:", list(pages.keys()))
pages[page_selection]()