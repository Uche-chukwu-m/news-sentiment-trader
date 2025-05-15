# src/tickers.py

TICKER_MAP = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN"
    # Add more here as needed
}

# Optional: list of display names
COMPANY_LIST = list(TICKER_MAP.keys())

# Optional: reverse lookup
TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}
