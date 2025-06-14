import yfinance as yf
import requests
from config import NEWS_API_KEY as API_KEY
from tickers import TICKER_MAP, COMPANY_LIST


# print(yf.download(("MSFT")))
url = f"https://newsapi.org/v2/everything?q={[company for company in COMPANY_LIST]}&apiKey={API_KEY}&language=en&sortBy=publishedAt&from=2025-05-01&to=2025-05-14"
response = requests.get(url)