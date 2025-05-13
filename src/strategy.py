import pandas as pd
import yfinance as yf
import os

def load_sentiment_data(path="data/news.csv"):
    df = pd.read_csv(path, parse_dates=["publishedAt"])
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["date"] = df["publishedAt"].dt.date  
    return df

def fetch_price_data(ticker, start,end):
    return yf.download(ticker, start=start, end=end)[["Close"]].reset_index()

def generate_signals(news_df):
    """Returns a DataFrame with one sentiment signal per day per company."""
    # Daily average sentiment
    sentiment_daily = news_df.groupby(["company", "date"])["sentiment"].mean().reset_index()
    def label(row):
        if row["sentiment"] > 0.2:
            return "buy"
        elif row["sentiment"] < -0.2:
            return "sell"
        else:
            return "hold"
    # Label each day with a signal
    sentiment_daily["signal"] = sentiment_daily.apply(label, axis=1)
    return sentiment_daily

if __name__ == "__main__":
    df_news = load_sentiment_data()
    signals = generate_signals(df_news)
    
    print(signals.head(10))