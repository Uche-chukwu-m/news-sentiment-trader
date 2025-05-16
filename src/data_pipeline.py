import requests
import csv
import os
from datetime import datetime, timedelta
from config import NEWS_API_KEY as API_KEY
from sentiment import analyze_sentiment
from tickers import TICKER_MAP, COMPANY_LIST
import time


# Initialize dates properly as datetime objects
today = datetime.today()
end_date = today
start_date = today - timedelta(days=7) # Fetch news for the last 7 days

def fetch_news(company, from_date, to_date):
    """Fetches news articles for a single company and writes them to a CSV."""
    # Format dates as strings for the API
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={API_KEY}&language=en&sortBy=publishedAt&from={from_date_str}&to={to_date_str}"
    response = requests.get(url)
    
    # Check if the response was successful
    if response.status_code != 200:
        print(f"Error fetching data for {company}: {response.status_code}")
        return []
    
    data = response.json()
    articles = data.get("articles", [])

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
                company,
                sentiment
            ])

    print(f"Fetched {len(articles)} articles for {company} from {from_date_str} to {to_date_str}")
    return articles

if __name__ == "__main__":
    companies = COMPANY_LIST
    all_articles = []
    
    # Process each day in the range 
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        
        for company in companies:
            news = fetch_news(company, current_date, next_date)
            all_articles.extend(news)
            time.sleep(1.5) 
        
        current_date = next_date
        
        # Optional: Add a small delay to avoid API rate limits
        time.sleep(1.5)  # Adjust the delay as needed   

    print(f"Total articles fetched: {len(all_articles)}")