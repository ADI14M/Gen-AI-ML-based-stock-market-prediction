import streamlit as st

st.set_page_config(page_title="StockVision AI - Stock Market Dashboard", layout="wide")

from dashboard_home import show_dashboard
from ipo_dashboard import show_ipo_dashboard
from ai_insights import show_ai_insights
from ai_assistant import show_ai_assistant
from twitter_sentiment_analysis import run_twitter_sentiment_analysis
from stock_sentiment_analysis import run_stock_sentiment_analysis
from stock_trend_prediction import run_stock_trend_prediction
from investors_info import show_investors
from analyst_ratings import show_analyst_ratings


st.title("📈 StockVision AI - Market Intelligence Dashboard")


pages = {
    "🏠 Home / Dashboard": show_dashboard,
    "🧠 AI StockVision Score": show_ai_insights,
    "🤖 AI Market Assistant": show_ai_assistant,
    "🚀 IPO Dashboard": show_ipo_dashboard,
    "💹 Stock Sentiment Analysis": run_stock_sentiment_analysis,   
    "🐦 Twitter Sentiment Analysis": run_twitter_sentiment_analysis,
    "📊 Stock Trend Prediction": run_stock_trend_prediction,
    "🏦 Institutional & Major Investors": show_investors,
    "📑 Analyst Ratings & Price Targets": show_analyst_ratings,
}


page_selection = st.sidebar.selectbox("🔍 Select a page:", list(pages.keys()))
pages[page_selection]()