# 📈 StockVision AI - Market Intelligence Dashboard

StockVision AI is an **AI-powered stock market analysis dashboard** built with **Streamlit, HuggingFace FinBERT, and ML models**.  
It provides insights into **sentiments, trends, investors, and analyst ratings** to help users make informed decisions.

---

## 🚀 Features

- 🐦 **Twitter Sentiment Analysis**  
  Analyze market mood based on tweets about a stock or market event.  

- 💹 **Stock Sentiment Analysis (FinBERT + Yahoo Finance)**  
  Uses **ProsusAI/FinBERT** to classify recent financial news articles into Positive, Negative, or Neutral sentiment.  

- 📊 **Stock Trend Prediction**  
  Predicts future stock trends for user-provided tickers using ML models.  

- 🏦 **Institutional & Major Investors**  
  Displays information about institutional holdings and key investors.  

- 📑 **Analyst Ratings & Price Targets**  
  View analyst predictions, buy/hold/sell ratings, and future price targets.  

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Interactive dashboard  
- [HuggingFace Transformers](https://huggingface.co/) – FinBERT sentiment analysis  
- [Feedparser](https://pythonhosted.org/feedparser/) – Yahoo Finance RSS feeds  
- [Pandas, Matplotlib] – Data visualization  
- [Scikit-learn / ML models] – Stock trend prediction  

---

## 📂 Project Structure

📦 StockVision-AI
├── app.py # Main dashboard (Streamlit)
├── twitter_sentiment_analysis.py
├── stock_sentiment_analysis.py
├── stock_trend_prediction.py
├── investors_info.py
├── analyst_ratings.py
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## ⚡ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ADI14M/StockVision-AI.git
   cd StockVision-AI
pip install -r requirements.txt

▶️ Running the App

Run the Streamlit dashboard:
streamlit run app.py

Then open your browser at http://localhost:8501 🎉

👨‍💻 Author

Aditya M