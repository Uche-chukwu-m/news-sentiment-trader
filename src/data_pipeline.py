import requests
import csv
import os
from datetime import datetime
from config import NEWS_API_KEY as API_KEY
from sentiment import analyze_sentiment

def fetch_news(ticker):
    """Fetches news articles for a single ticker and writes them to a CSV."""
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}&language=en&sortBy=publishedAt"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    with open("data/news.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header only if file is empty
        if f.tell() == 0:
            writer.writerow(["publishedAt", "source", "title", "url", "company", "sentiment"])

        for article in articles:
            sentiment = analyze_sentiment(article.get("title", ""))
            writer.writerow([
                article.get("publishedAt", ""),
                article.get("source", {}).get("name", ""),
                article.get("title", ""),
                article.get("url", ""),
                ticker,
                sentiment
            ])

    print(f"Fetched {len(articles)} articles for {ticker}")
    return articles

if __name__ == "__main__":
    tickers = ["Apple", "Tesla", "Microsoft", "Amazon"]
    all_articles = []

    for ticker in tickers:
        news = fetch_news(ticker)
        all_articles.extend(news)

    print(f"Total articles fetched: {len(all_articles)}")
