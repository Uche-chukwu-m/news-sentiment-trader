import pandas as pd
import yfinance as yf
import os

# Company name to Yahoo Finance ticker symbol
TICKER_MAP = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN"
    # Add more if needed
}

def load_sentiment_data(path="data/news.csv"):
    df = pd.read_csv(path, parse_dates=["publishedAt"])
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["date"] = df["publishedAt"].dt.date  
    return df

def fetch_price_data(ticker, start, end):
    """Fetches price data from Yahoo Finance and returns a DataFrame with date and Close price."""
    # Add buffer days to ensure we have enough data for next-day calculations

    start_date = pd.to_datetime(start) - pd.Timedelta(days=2)
    end_date = pd.to_datetime(end) + pd.Timedelta(days=2)

    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data for {ticker} in the specified date range.")
        return pd.DataFrame()
    
    # Handle multi-index columns (common with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    # Reset index to make "Date" a column
    df = df.reset_index()

    # Ensure we have the Close price column
    if "Close" not in df.columns:
        print(f"⚠️ Couldn't find Close column in {ticker}'s data")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Select only the "Date" and "Close" columns
    df = df[["Date", "Close"]]
    
    # Convert "Date" column to datetime if it's not already
    df["date"] = df["Date"].dt.date

    print(f"Downloaded data for {ticker}:")
    print("Result shape:", df.shape)
    print("Downloaded columns:", df.columns)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df[["date", "Close"]]


def generate_signals(news_df):
    """Returns a DataFrame with one sentiment signal per day per company."""
    # Standardize company names
    news_df["company"] = news_df["company"].str.title().str.strip()

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

def merge_with_prices(signals_df):
    """Merges sentiment signals with price data."""
    result = []
    companies = signals_df["company"].unique()

    for company in companies:
        signals_df["company"] = signals_df["company"].str.title().str.strip()
        df_signal = signals_df[signals_df["company"] == company].copy()
        start = "2025-05-07" #pd.to_datetime(df_signal["date"].min()) - pd.Timedelta(days=5)
        end = "2025-05-13" #pd.to_datetime(df_signal["date"].max()) + pd.Timedelta(days=5)


        ticker = TICKER_MAP.get(company)
        
        # Check if ticker is None or empty
        if not ticker:
            print(f"⚠️ No ticker mapping for {company}, skipping.")
            continue

        # Fetch price data
        try:
            price_df = fetch_price_data(ticker, str(start), str(end))
            if price_df.empty:
                print(f"⚠️ No price data for {ticker}, skipping.")
                continue
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            continue

        # Merge signals with price data
        merged = pd.merge(df_signal, price_df, on="date", how="left")
        merged["company"] = company
        result.append(merged)

    # Concatenate all results
    if not result:
        return pd.DataFrame()  
     
    return pd.concat(result)

def simulate_returns(df):
    """Simulates next-day returns based on sentiment signals."""
    if df.empty:
        return df
    
    # Sort by company and date
    df = df.sort_values(["company", "date"]).copy()
    df["next_close"] = df.groupby("company")["Close"].shift(-1)
    df["return"] = (df["next_close"] - df["Close"]) / df["Close"]
    # above line uses the return formula: (price tomorrow - price today)/price today
    
    def apply_trade(row):
        if row["signal"] == "buy":
            return row["return"]
        elif row["signal"] == "sell":
            return -row["return"]
        else:
            return 0
        
    df["strategy_return"] = df.apply(apply_trade, axis=1)
    df["cumulative_return"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    return df


if __name__ == "__main__":
    df_news = load_sentiment_data()
    signals = generate_signals(df_news)

    merged = merge_with_prices(signals)
    result = simulate_returns(merged)

    print(signals.head(10))
    print(result.head(10))
    print("\nSignals data sample:")
    print(signals.head(5))
    
    print("\nStrategy results sample:")
    if not result.empty:
        print(result[["company", "date", "sentiment", "signal", "Close", "next_close", "return", "strategy_return", "cumulative_return"]].head(5))
    else:
        print("No results generated. Check for errors above.")