from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Returns the compound sentiment score for a given text"""
    if not text:
        return 0.0
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores