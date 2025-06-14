import matplotlib.pyplot as plt
import pandas as pd
import os
from src.tickers import COMPANY_LIST # or TICKER_MAP.keys()


def load_data(file_path="data/news.csv"):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}. Run the data_pipeline first.')
    df = pd.read_csv(file_path, parse_dates=['publishedAt'])
    df["sentiment"] = pd.to_numeric(df["sentiment"].astype(float), errors='coerce')
    return df

def plot_avg_sentiment_per_company(df):
    avg_sentiment = df.groupby('company')['sentiment'].mean().sort_values()
    avg_sentiment.plot(kind='bar', figsize=(10, 6), title='Average Sentiment per Company')
    plt.ylabel('Average Sentiment Score')
    plt.xlabel('Company')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_sentiment_trend(df, company):
    df_company = df[df['company'] == company].copy()
    df_company["date"] = df_company["publishedAt"].dt.date
    daily_sentiment = df_company.groupby("date")["sentiment"].mean()
    daily_sentiment.plot(figsize=(10, 6), title=f'Sentiment Trend for {company} Over Time', marker='o')
    plt.ylabel('Average Daily Sentiment')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    plot_avg_sentiment_per_company(df)
    for company_name in COMPANY_LIST: # Use the imported list
        plot_sentiment_trend(df, company_name)
