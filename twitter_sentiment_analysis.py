import streamlit as st
from transformers import pipeline
import pandas as pd
import random
import time

# Initialize the Hugging Face sentiment analysis pipeline
# Check for GPU availability for faster processing (optional)
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

sentiment_pipeline = load_sentiment_pipeline()

def get_mock_tweets(topic, count=10):
    """
    Generate mock tweets for demonstration purposes when API keys are not available.
    """
    templates = [
        f"{topic} is performing really well today! Stock is up.",
        f"I'm worried about {topic}'s recent earnings report.",
        f"Just bought more shares of {topic}. Bullish!",
        f"{topic} management needs to step up their game.",
        f"Market sentiment for {topic} seems mixed right now.",
        f"Can't believe {topic} dropped this much...",
        f"{topic} has great potential for long-term growth.",
        f"Anyone else watching {topic} closely?",
        f"Technicals for {topic} look bearish.",
        f"Huge announcement coming from {topic} soon?"
    ]
    
    tweets = []
    for _ in range(count):
        tweets.append(random.choice(templates))
    return tweets

def run_twitter_sentiment_analysis():
    st.header("🐦 Twitter Sentiment Analysis")
    st.write("Analyze public sentiment from Twitter (simulated) for specific stocks or topics.")

    # Input for topic
    topic = st.text_input("Hashtag or Topic to Search (e.g., Apple, Tesla, Bitcoin):", "Apple")
    
    # Slider for number of tweets
    count = st.slider("Number of tweets to analyze:", min_value=5, max_value=50, value=10)

    if st.button("Analyze Sentiment"):
        with st.spinner(f"Fetching and analyzing tweets for '{topic}'..."):
            # Simulate network delay for realism
            time.sleep(1.5)
            
            # Use mock tweets (since we don't have active Twitter API keys configured)
            tweets = get_mock_tweets(topic, count)
            
            # Analyze sentiment
            results = []
            positive_count = 0
            negative_count = 0
            
            for tweet in tweets:
                analysis = sentiment_pipeline(tweet)[0]
                label = analysis["label"]
                score = float(analysis["score"])
                
                results.append({
                    "Tweet": tweet,
                    "Sentiment": label,
                    "Confidence": round(score, 4)
                })
                
                if label == "POSITIVE":
                    positive_count += 1
                else:
                    negative_count += 1
            
            df = pd.DataFrame(results)
            
            # Display Summary Metrics
            st.subheader("📊 Sentiment Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tweets", count)
            col2.metric("Positive Tweets", positive_count, delta_color="normal")
            col3.metric("Negative Tweets", negative_count, delta_color="normal")

            # Display Data
            st.subheader("📝 Detailed Analysis")
            st.dataframe(df, use_container_width=True)
            
            # Simple Bar Chart
            st.subheader("📈 Visualization")
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [positive_count, negative_count]
            })
            st.bar_chart(chart_data.set_index("Sentiment"))

if __name__ == "__main__":
    # Test run
    run_twitter_sentiment_analysis()
