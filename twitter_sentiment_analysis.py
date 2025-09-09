from transformers import pipeline

# Initialize the Hugging Face sentiment analysis pipeline with PyTorch
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"   # 👈 Force PyTorch, avoid TensorFlow errors
)

def run_twitter_sentiment_analysis(tweets):
    """
    Run sentiment analysis on a list of tweets.

    Args:
        tweets (list[str]): List of tweet texts.

    Returns:
        list[dict]: Each dict contains 'label' and 'score' for the tweet.
    """
    results = []
    for tweet in tweets:
        analysis = sentiment_pipeline(tweet)[0]  # pipeline returns a list
        results.append({
            "tweet": tweet,
            "label": analysis["label"],
            "score": float(analysis["score"])
        })
    return results


# Example run (you can remove this in production)
if __name__ == "__main__":
    sample_tweets = [
        "The stock market is looking great today!",
        "I lost so much money in stocks, feeling terrible..."
    ]
    output = run_twitter_sentiment_analysis(sample_tweets)
    for res in output:
        print(res)
