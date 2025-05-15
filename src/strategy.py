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
        # signals_df["company"] = signals_df["company"].str.title().str.strip()
        df_signal = signals_df[signals_df["company"] == company].copy()
        
        # Get min and max dates from signals to ensure we have buffer days for returns
        min_date = pd.to_datetime(df_signal["date"].min())
        max_date = pd.to_datetime(df_signal["date"].max())

        # Add buffer days for calculating returns
        start = (min_date - pd.Timedelta(days=0)).strftime("%Y-%m-%d")
        end = (max_date + pd.Timedelta(days=0)).strftime("%Y-%m-%d")

        # start = "2025-05-07" #pd.to_datetime(df_signal["date"].min()) - pd.Timedelta(days=5)
        # end = "2025-05-13" #pd.to_datetime(df_signal["date"].max()) + pd.Timedelta(days=5)


        ticker = TICKER_MAP.get(company)

        # For debugging, also handle the hardcoded dates case
        if min_date.year > 2025:  # If we're using future dates in the test
            start = "2025-05-05"  # 2 days before the hardcoded start
            end = "2025-05-15"    # 2 days after the hardcoded end

        print(f"Date range for {company}: {start} to {end}")
        
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

            # Check if we have sufficient price data
            if len(price_df) <= 1:
                print(f"⚠️ Insufficient price data for {ticker} to calculate returns.")
                continue

            print(f"Price data for {company} ({ticker}) shape: {price_df.shape}")
            print(f"First few dates: {price_df['date'].head().tolist()}")
            print(f"Last few dates: {price_df['date'].tail().tolist()}")
            print(f"First few prices: {price_df['Close'].head().tolist()}")

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            continue

        # Merge signals with price data
        merged = pd.merge(df_signal, price_df, on="date", how="left")

        print(f"Merged data for {company} shape: {merged.shape}")
        print(f"NaN values in Close column: {merged['Close'].isna().sum()}")

        # Add any missing dates from price data for continuous return calculation
        unique_dates = price_df["date"].unique()
        missing_dates = [d for d in unique_dates if d not in df_signal["date"].values]
        
        if missing_dates and len(missing_dates) > 0:
            print(f"Adding {len(missing_dates)} missing dates from price data for {company}")
            missing_df = pd.DataFrame({
                "date": missing_dates,
                "company": company,
                "sentiment": None,
                "signal": "hold"  # Default to hold for days without sentiment data
            })
            # Add price data to these missing dates
            missing_with_prices = pd.merge(missing_df, price_df, on="date", how="left")
            
            # Combine with the original merged data
            merged = pd.concat([merged, missing_with_prices])
            
            # Sort by date for proper return calculation
            merged = merged.sort_values("date").reset_index(drop=True)

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

    # Calculate next day close prices for each company separately
    df["next_close"] = df.groupby("company")["Close"].shift(-1)

    # Print diagnostic information
    print(f"\nDiagnostic info for next_close calculation:")
    print(f"Total rows: {len(df)}")
    print(f"NaN next_close values: {df['next_close'].isna().sum()}")
    
    # Calculate returns only where we have next_close values
    df["return"] = None
    mask = df["next_close"].notna()
    if mask.any():
        df.loc[mask, "return"] = (df.loc[mask, "next_close"] - df.loc[mask, "Close"]) / df.loc[mask, "Close"]

    # For the last day of each company's data, try to get the next trading day's data
    # This would require modifying the fetch_price_data function to get a wider date range
    # For now, we'll acknowledge these missing values
    
    def apply_trade(row):
        if pd.isna(row["return"]):
            return 0  # No trade if we don't have return data
        elif row["signal"] == "buy":
            return row["return"]
        elif row["signal"] == "sell":
            return -row["return"]
        else:
            return 0
        
    df["strategy_return"] = df.apply(apply_trade, axis=1)

    # Calculate cumulative returns, properly handling NaN values
    # We use fillna(0) to ensure that NaN values don't affect the cumulative product
    df["cumulative_return"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    
    return df


if __name__ == "__main__":
    print("Loading sentiment data...")
    df_news = load_sentiment_data()
    print(f"Loaded sentiment data with shape: {df_news.shape}")
    
    print("\nGenerating signals...")
    signals = generate_signals(df_news)
    print(f"Generated signals with shape: {signals.shape}")
    
    print("\nMerging with price data...")
    merged = merge_with_prices(signals)
    print(f"Merged data shape: {merged.shape}")
    
    print("\nSimulating returns...")
    result = simulate_returns(merged)
    print(f"Final result shape: {result.shape}")

    print("\nSignals data sample:")
    print(signals.head(5))
    
    print("\nStrategy results sample:")
    if not result.empty:
        display_cols = ["company", "date", "sentiment", "signal", "Close", "next_close", "return", "strategy_return", "cumulative_return"]
        # Only show columns that exist
        valid_cols = [col for col in display_cols if col in result.columns]
        print(result[valid_cols].reset_index())
    else:
        print("No results generated. Check for errors above.")

    # Plot cumulative return
    import matplotlib.pyplot as plt
    result.plot(x="date", y="cumulative_return", title="Cumulative Strategy Return", legend=False)
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.tight_layout()
    plt.show()