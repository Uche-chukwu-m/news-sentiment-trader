# 📰 News Sentiment Trading Bot

This project implements and analyzes a trading strategy driven by financial news sentiment. It fetches news articles for specified companies, scores their sentiment using VADER, generates daily buy/sell/hold signals, and backtests these signals against historical stock price data from Yahoo Finance. The results, including cumulative returns and comparisons to a buy-and-hold strategy, are visualized in an interactive Streamlit dashboard.

---

## Features

- **Sentiment Analysis** from financial news headlines
- **Buy/Sell/Hold Signal Generation** per company per day
- **Price Fetching** via Yahoo Finance (`yfinance`)
- **Backtesting** of strategy with 1-day returns
- **Cumulative Return Visualization**
- **Buy-and-Hold vs Strategy Performance Comparison**
- **Streamlit Dashboard (coming soon)**

---


# Timeline


## Project Setup + Data Ingestion

- ✅ Created folder structure (`src/`, `data/`, `app.py`)
- ✅ Set up `config.py` to safely store NewsAPI key
- ✅ Wrote initial `fetch_news()` using NewsAPI
- ✅ Enabled multi-ticker support (Apple, Tesla, Microsoft, Amazon)
- ✅ Appended all articles to `data/news.csv`
- ✅ Added `company` and `sentiment` fields for each row

## Sentiment Analysis + Visualization

- ✅ Used VADER to score each headline (`compound` score only)
- ✅ Appended sentiment values to `news.csv`
- ✅ Created `generate_signals()` to label each day:
  - `buy` if sentiment > 0.2
  - `sell` if sentiment < -0.2
  - `hold` otherwise
- ✅ Built bar chart for average sentiment per company
- ✅ Added `src/tickers.py` to modularize company-to-ticker mapping

## Signal-Price Merging + Simulation

- ✅ Wrote `fetch_price_data()` using `yfinance`
- ✅ Merged sentiment signals with price data on matching dates
- ✅ Calculated next-day returns
- ✅ Simulated returns for:
  - Buy → gain if price rises
  - Sell → gain if price falls
  - Hold → no trade
- ✅ Tracked cumulative return over time
- ✅ Plotted company-wise and portfolio-wise strategy performance
- ✅ Added buy/sell markers to plots

## Strategy Evaluation + Automation Plan

- ✅ Compared sentiment strategy to buy-and-hold returns
- ✅ Logged best and worst daily returns
- ✅ Set up `daily_fetch.py` for automated news pulling
- ✅ Scheduled it using **Windows Task Scheduler**
- ✅ Finalized full pipeline with:
  - Sentiment scoring
  - Signal generation
  - Price merging
  - Simulation
  - Visualization

## Streamlit Development & Refinement (Ongoing)

- ✅ Enhanced app.py (Streamlit Dashboard):
  - ✅ Integrated data_pipeline.py and strategy.py to run the full workflow from the UI.
  - ✅ Display of cumulative return plots (individual and portfolio) using matplotlib via st.pyplot().
  - ✅ Tabular display of detailed strategy results and performance summaries.
  - ✅ Visualization of raw news data and sentiment trends (average sentiment per company, daily sentiment trend for selected company).
  - ✅ Implemented robust data handling to prevent stale data issues across multiple runs/selections by clearing and reloading data specific to user inputs.
  - ✅ Added debug outputs and error handling for a better user experience.
- ✅ Refined import paths and project structure for better modularity (e.g., src as a package).
- ✅ Addressed and fixed bugs related to data consistency, merging, and Streamlit caching.

---

## 📌 Files of Interest

| File                  | Purpose |
|-----------------------|---------|
| `data_pipeline.py`    | News fetching and sentiment scoring |
| `tickers.py`          | Company to ticker mapping |
| `strategy.py`         | Signal generation and backtesting |
| `daily_fetch.py`      | Script to run daily via Task Scheduler |
| `app.py`              | Streamlit UI (coming soon) |

---
