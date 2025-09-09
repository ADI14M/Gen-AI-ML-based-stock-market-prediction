import streamlit as st
from transformers import pipeline
import feedparser

# Load FinBERT pipeline (force PyTorch to avoid TensorFlow issues)
pipe = pipeline(task="text-classification", model="ProsusAI/finbert", framework="pt")

def analyze_sentiment(ticker, keyword):
    """
    Fetch Yahoo Finance RSS feed for the given ticker
    and analyze sentiment of articles containing the keyword.
    """
    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    articles = []

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue

        sentiment = pipe(entry.summary)[0]
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'summary': entry.summary,
            'sentiment': sentiment['label'],
            'score': sentiment['score']
        })

        if sentiment['label'].lower() == 'positive':
            total_score += sentiment['score']
            num_articles += 1
        elif sentiment['label'].lower() == 'negative':
            total_score -= sentiment['score']
            num_articles += 1

    # Calculate average sentiment score
    if num_articles > 0:
        final_score = total_score / num_articles
    else:
        final_score = 0

    # Determine overall sentiment
    if final_score > 0.2:
        overall_sentiment = "Positive"
    elif final_score < -0.2:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return overall_sentiment, final_score, articles


def run_stock_sentiment_analysis():
    st.header("💹 Stock Sentiment Analysis (FinBERT + Yahoo Finance)")
    st.write("Enter a stock ticker and optional keyword to analyze recent financial news sentiment.")

    ticker_input = st.text_input("🔎 Stock Ticker (e.g., AAPL, TSLA, INFY, RELIANCE):", "AAPL")
    keyword_input = st.text_input("📌 Keyword to filter articles (optional):", "")

    if st.button("Analyze Sentiment"):
        if ticker_input:
            with st.spinner("Fetching and analyzing news..."):
                overall_sentiment, final_score, articles = analyze_sentiment(ticker_input, keyword_input or ticker_input)
            
            # Show overall sentiment
            st.subheader("📈 Overall Sentiment")
            st.write(f"**Sentiment:** {overall_sentiment}  |  **Score:** {final_score:.2f}")

            # Show article-level details
            if articles:
                st.subheader("📰 Analyzed Articles")
                for article in articles:
                    st.markdown(f"**Title:** [{article['title']}]({article['link']})")
                    st.write(f"**Published:** {article['published']}")
                    st.write(f"**Summary:** {article['summary']}")
                    st.write(f"**Sentiment:** {article['sentiment']}  |  **Score:** {article['score']:.2f}")
                    st.write("---")
            else:
                st.warning("No matching articles found for the given keyword.")
        else:
            st.error("Please enter a valid stock ticker symbol.")
