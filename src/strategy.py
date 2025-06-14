import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.tickers import TICKER_MAP, COMPANY_LIST

"""
This strategy assumes that a trade is entered at the closing price of the day the signal is generated (Day T).
The trade is then exited at the closing price of the next trading day (Day T+1).
This is a very short-term, overnight holding strategy.
"""

"""
For each day, take the strategy_return (e.g., 0.02 for a 2% gain).
Add 1 to it (e.g., 1.02). This is the "growth factor."
For each company, multiply twhese daily growth factors together sequentially over time.
This shows how an initial investment (e.g., $1) would have grown or shrunk if you followed the strategy's signals day by day.
For example:
Day 1: Signal Buy, Stock up 2% -> strategy_return = 0.02. Cumulative = 1.02
Day 2: Signal Sell, Stock down 1% -> strategy_return = 0.01. Cumulative = 1.02 * (1 + 0.01) = 1.02 * 1.01 = 1.0302
Day 3: Signal Hold -> strategy_return = 0.00. Cumulative = 1.0302 * (1 + 0.00) = 1.0302

"""
def load_sentiment_data(path="data/news.csv"):
    df = pd.read_csv(path, parse_dates=["publishedAt"])
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["date"] = df["publishedAt"].dt.date  
    return df

def fetch_price_data(ticker, start, end):
    """Fetches price data from Yahoo Finance and returns a DataFrame with date and Close price."""
    # Add buffer days to ensure we have enough data for next-day calculations

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end) + pd.Timedelta(days=5) # 5 days buffer for weekends/holidays

    # fetch_start_date = start_date - pd.Timedelta(days=5) # Increased buffer
    # fetch_end_date = end_date + pd.Timedelta(days=5)   # Increased buffer

    print(f"--- DEBUG: fetch_price_data for {ticker} ---")
    print(f"Original signal date range: {start} to {end}")
    print(f"Fetching yfinance data from: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data for {ticker} in the specified date range.")
        return pd.DataFrame()
    
    print(f"Raw yfinance data shape for {ticker}: {df.shape}")
    
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
    threshold = 0.1  # Threshold for sentiment to generate buy/sell signals

    def label(row):
        if row["sentiment"] > threshold:
            return "buy"
        elif row["sentiment"] < threshold * -1:
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

        company = company.title().strip()
        ticker = TICKER_MAP.get(company, company)

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
    df["cumulative_return"] = (1 + df["strategy_return"].fillna(0)).groupby(df["company"]).cumprod()

    return df

def plot_cumulative_returns(result_df, save_path=None, return_fig=False):
    """
    Plot cumulative returns over time for each company.
    
    Args:
        result_df: DataFrame with cumulative returns
        save_path: If provided, save the plot to this file instead of displaying
        return_fig: If True, return the figure object instead of showing/saving it
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    if result_df.empty:
        print("No data to plot.")
        return
    
    # Convert date to datetime for better plotting
    result_df["date_dt"] = pd.to_datetime(result_df["date"])
    
    # Create a figure with proper size
    plt.figure(figsize=(12, 7))
    
    # Plot each company separately
    companies = result_df["company"].unique()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, company in enumerate(companies):
        company_data = result_df[result_df["company"] == company].sort_values("date_dt")
        if not company_data.empty and "cumulative_return" in company_data.columns:
            color = colors[i % len(colors)]
            
            # Get previous day's cumulative return for legend
            legend_return = company_data['cumulative_return'].iloc[-2]
                
            plt.plot(company_data["date_dt"], company_data["cumulative_return"], 
                     label=f"{company} ({legend_return:.2f})",
                     color=color, linewidth=2)
            
            # Mark only the most recent buy and sell signals for each company
            # Find the most recent buy signal
            buys = company_data[company_data["signal"] == "buy"]
            sells = company_data[company_data["signal"] == "sell"]
            plt.scatter(buys["date_dt"], buys["cumulative_return"], color='green', marker='^', s=50, label=f"{company} Buy" if i == 0 and not buys.empty else None)
            plt.scatter(sells["date_dt"], sells["cumulative_return"], color='red', marker='v', s=50, label=f"{company} Sell" if i == 0 and not sells.empty else None)
            # Remove the 'if i == 0' if you want labels for each company's markers
    
    # Plot combined portfolio (if we want an equally weighted portfolio of all signals)
    if len(companies) > 1:
        # Group by date and calculate mean cumulative return across all companies
        portfolio = result_df.groupby("date_dt")["cumulative_return"].mean().reset_index()
        
        # Get latest portfolio return for legend
        portfolio_legend_return = portfolio['cumulative_return'].iloc[-1]
            
        plt.plot(portfolio["date_dt"], portfolio["cumulative_return"], 
                 label=f"Portfolio ({portfolio_legend_return:.2f})",
                 color='black', linewidth=3, linestyle='--')
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    # Add grid, labels, and legend
    plt.grid(True, alpha=0.3)
    plt.title("Cumulative Strategy Return by Company", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.3)  # Initial investment line
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Add annotations for final returns
    for i, company in enumerate(companies):
        company_data = result_df[result_df["company"] == company]
        if not company_data.empty:
            final_return = company_data["cumulative_return"].iloc[-2]
            latest_date = company_data["date_dt"].iloc[-2]
            plt.annotate(f"{final_return:.2f}", 
                         (latest_date, final_return),
                         xytext=(5, 0), textcoords='offset points', 
                         fontsize=10, fontweight='bold')
    
    if return_fig:
        fig = plt.gcf() # Get current figure
        plt.close(fig) # Close the figure to prevent it from displaying in non-Streamlit contexts if not intended
        return fig
    elif save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close() # Close after saving
    else:
        plt.show() # This will block if run directly
        plt.close() # Close after showing
    return None # If not returning fig


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
    print(signals)
    
    print("\nStrategy results sample:")
    if not result.empty:
        display_cols = ["company", "date", "sentiment", "signal", "Close", "next_close", "return", "strategy_return", "cumulative_return"]
        # Only show columns that exist
        valid_cols = [col for col in display_cols if col in result.columns]
        print(result[valid_cols].reset_index())

         # Show summary statistics
        print("\nSummary by company:")
        for company in result["company"].unique():
            company_data = result[result["company"] == company]
            last_cr = company_data["cumulative_return"].iloc[-2] if not company_data.empty else None
            nan_returns = company_data["return"].isna().sum()
            print(f"{company}: Final CR: {last_cr:.4f}, Missing returns: {nan_returns}/{len(company_data)}")
        
        # Plot cumulative returns
        print("\nPlotting cumulative returns...")
        plot_cumulative_returns(result)
        
        # Visualize the trading signals and their performance
        print("\nPlotting signals and performance...")
        # First, identify best and worst performing days
        result["daily_return"] = result["strategy_return"]
        best_days = result.sort_values("daily_return", ascending=False).head(5)
        worst_days = result.sort_values("daily_return").head(5)
        
        print("\nBest performing days:")
        print(best_days[["company", "date", "signal", "daily_return"]])
        
        print("\nWorst performing days:")
        print(worst_days[["company", "date", "signal", "daily_return"]])
        
        # Additional plot: Compare strategy vs buy-and-hold
        # For each company, calculate buy-and-hold return
        print("\nComparing strategy vs buy-and-hold...")
        for company in result["company"].unique():
            company_data = result[result["company"] == company].copy()
            if len(company_data) > 1:
                # Calculate buy and hold return
                first_price = company_data["Close"].iloc[0]
                last_price = company_data["Close"].iloc[-2]
                bh_return = (last_price / first_price) - 1  # Percentage return
                
                strategy_return = company_data["cumulative_return"].iloc[-2] - 1
                
                print(f"{company}: Strategy: {strategy_return:.2%}, Buy-and-Hold: {bh_return:.2%}, Difference: {strategy_return - bh_return:.2%}")
    else:
        print("No results generated. Check for errors above.")
