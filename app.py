import sys
import os

# Add the project root to sys.path
# This ensures that modules inside 'src' can find modules in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# --- Import your existing modules ---
import src.data_pipeline as data_pipeline
import src.strategy as strategy

# import src.visualize as visualize 
# Assuming visualize.py functions might be adapted or plots recreated directly

from src.tickers import COMPANY_LIST, TICKER_MAP
from src.config import NEWS_API_KEY

# --- App Configuration & Title ---
st.set_page_config(layout="wide", page_title="Sentiment Trading Strategy Analyzer")
st.title("📰 News Sentiment Trading Strategy Analyzer")
st.markdown("""
    This application fetches news articles, analyzes their sentiment,
    generates trading signals, and simulates the performance of a
    sentiment-based trading strategy.
""")

# --- Global Variables / Session State ---
# Initialize if not present, their content will be managed by the button logic
if 'news_data' not in st.session_state: # This will hold data for the current strategy run
    st.session_state.news_data = pd.DataFrame()
if 'strategy_results' not in st.session_state:
    st.session_state.strategy_results = pd.DataFrame()
if 'signals_data' not in st.session_state:
    st.session_state.signals_data = pd.DataFrame()

# --- Helper Functions (Specific to Streamlit interaction) ---
@st.cache_data # Cache based on filepath; if file content changes, it re-runs
def load_and_prepare_news_from_file(filepath="data/news.csv"):
    # st.write(f"DEBUG: `load_and_prepare_news_from_file` called for {filepath}") # Optional debug
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, parse_dates=["publishedAt"])
            if "publishedAt" in df.columns:
                df["date"] = df["publishedAt"].dt.date
            else:
                st.warning("'publishedAt' column not found in news.csv. Cannot create 'date' column.")
            if "sentiment" in df.columns:
                df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
            else:
                st.warning("'sentiment' column not found in news.csv.")
            return df
        except Exception as e:
            st.error(f"Error in `load_and_prepare_news_from_file` for {filepath}: {e}")
            return pd.DataFrame()
    # st.write(f"DEBUG: File {filepath} does not exist in `load_and_prepare_news_from_file`") # Optional debug
    return pd.DataFrame()

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")

# Date Range Selection
st.sidebar.subheader("📅 Date Range for Analysis")
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=7)

start_date_input = st.sidebar.date_input("Start Date", value=default_start_date, max_value=default_end_date - timedelta(days=1)) # Prevent end < start
end_date_input = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date_input + timedelta(days=1), max_value=datetime.today())

# Company Selection
st.sidebar.subheader("🏢 Company Selection")
available_companies = ["All Companies"] + COMPANY_LIST
selected_companies_option = st.sidebar.multiselect(
    "Select Companies",
    options=available_companies,
    default=["All Companies"]
)

if "All Companies" in selected_companies_option or not selected_companies_option:
    companies_to_process = COMPANY_LIST
else:
    companies_to_process = selected_companies_option

# --- Main Application Logic ---
st.sidebar.subheader("🚀 Actions")
fetch_button_disabled = not NEWS_API_KEY
if st.sidebar.button("Fetch Latest News & Run Strategy", disabled=fetch_button_disabled, key="fetch_run_button"):
    if not NEWS_API_KEY:
        st.error("NEWS_API_KEY is not configured. Cannot fetch news.")
    else:
        # --- 0. Clear Previous State for this new run ---
        st.session_state.news_data = pd.DataFrame()
        st.session_state.signals_data = pd.DataFrame()
        st.session_state.strategy_results = pd.DataFrame()
        # The cache for load_and_prepare_news_from_file will invalidate if news.csv changes.

        news_file_path = "data/news.csv"

        # --- 1. Data Fetching ---
        with st.spinner(f"Fetching news for: {', '.join(companies_to_process)}... This may take a while."):
            if os.path.exists(news_file_path):
                try:
                    os.remove(news_file_path)
                    st.info(f"Cleared existing {news_file_path} for a fresh data fetch.")
                except OSError as e:
                    st.warning(f"Could not remove existing {news_file_path}: {e}. This might lead to issues if fetch_news appends unexpectedly.")

            # total_articles_fetched_this_run = 0 # Using a counter instead of extending a list directly if fetch_news writes to file
            pipeline_start_date = datetime.combine(start_date_input, datetime.min.time())
            pipeline_end_date = datetime.combine(end_date_input, datetime.max.time())

            # current_fetch_date = pipeline_start_date
            # while current_fetch_date <= pipeline_end_date:
            # st.write(f"Fetching for date: {current_fetch_date.strftime('%Y-%m-%d')}") # Progress update
            for company in companies_to_process:
                # Assuming data_pipeline.fetch_news writes to news.csv and returns the articles fetched in that call
                articles_in_call = data_pipeline.fetch_news(company, pipeline_start_date, pipeline_end_date)
            #     if articles_in_call: # If fetch_news returns a list/count
            #         total_articles_fetched_this_run += len(articles_in_call)
            # current_fetch_date += timedelta(days=1) # Increment date

            st.success(f"News fetching process complete. Articles logged to {news_file_path}.") # Adjusted message

            # --- Load the freshly created/overwritten news.csv ---
            # This data should ONLY be for the current companies_to_process and date range
            df_loaded_news = load_and_prepare_news_from_file(news_file_path)

            if df_loaded_news.empty:
                st.warning(f"No news data loaded from {news_file_path} after fetching. Cannot proceed with strategy.")
            else:
                # Filter again as a safeguard, though ideally news.csv is clean
                df_filtered_for_strategy = df_loaded_news[df_loaded_news['company'].isin(companies_to_process)].copy()
                # if df_filtered_for_strategy.empty and not df_loaded_news.empty :
                    # st.warning(f"Data was loaded from {news_file_path}, but after filtering for selected companies ({', '.join(companies_to_process)}), no data remains.")
                st.session_state.news_data = df_filtered_for_strategy # Store the correctly scoped data
                st.info(f"Successfully prepared {len(st.session_state.news_data)} news items for the strategy.")

        # --- 2. Strategy Simulation (only if st.session_state.news_data is populated) ---
        if not st.session_state.news_data.empty:
            with st.spinner("Generating signals and simulating strategy..."):
                df_for_signals = st.session_state.news_data # Use the cleaned, session-specific data

                # --- DEBUG block for generate_signals input ---
                st.markdown("--- DEBUG: Input to `generate_signals` ---")
                st.write("Columns:", df_for_signals.columns.tolist())
                st.dataframe(df_for_signals.head())
                if 'date' not in df_for_signals.columns:
                    st.error("CRITICAL: 'date' column MISSING before `generate_signals`!")
                
                # --- End DEBUG ---

                st.session_state.signals_data = strategy.generate_signals(df_for_signals)

                # --- DEBUG block for merge_with_prices input ---
                st.markdown("--- DEBUG: Input to `merge_with_prices` ---")
                if st.session_state.signals_data.empty:
                    st.warning("`signals_data` is empty.")
                else:
                    st.write("`signals_data` Columns:", st.session_state.signals_data.columns.tolist())
                    st.dataframe(st.session_state.signals_data.head())
                # --- End DEBUG ---

                merged_data = strategy.merge_with_prices(st.session_state.signals_data)

                # --- DEBUG block for simulate_returns input ---
                st.markdown("--- DEBUG: Input to `simulate_returns` ---")
                if merged_data.empty:
                    st.warning("`merged_data` is empty.")
                else:
                    st.write("`merged_data` Columns:", merged_data.columns.tolist())
                    st.dataframe(merged_data.head())
                    st.write(f"NaNs in `merged_data['Close']`: {merged_data['Close'].isna().sum()} / {len(merged_data)}")
                # --- End DEBUG ---

                if 'date' not in df_for_signals.columns:
                    st.error("CRITICAL: 'date' column is missing from the loaded news data. Cannot generate signals.")
                else:
                    st.session_state.signals_data = strategy.generate_signals(df_for_signals)
                    merged_data = strategy.merge_with_prices(st.session_state.signals_data)
    
                if not merged_data.empty:
                    st.session_state.strategy_results = strategy.simulate_returns(merged_data)
                    st.success("Strategy simulation complete!")
                else:
                    st.warning("No price data could be merged or strategy could not be simulated (merged_data was empty).")
                    st.session_state.strategy_results = pd.DataFrame() # Ensure cleared
        else:
            # This 'else' corresponds to st.session_state.news_data.empty after fetching and loading attempt
            if NEWS_API_KEY: # Only show if fetching was actually attempted
                st.warning("No news data was available after the fetching and loading process to run the strategy.")


# --- Displaying Results ---
# These sections will now use the st.session_state variables which were correctly populated or cleared by the button logic.
st.header("📊 Strategy Performance & Analysis")

if not st.session_state.strategy_results.empty:
    results_df_display = st.session_state.strategy_results # This is already filtered by the companies processed in the last run

    st.subheader("📈 Cumulative Strategy Returns")
    fig_cumulative_returns = strategy.plot_cumulative_returns(results_df_display, return_fig=True)
    if fig_cumulative_returns:
        st.pyplot(fig_cumulative_returns)
    else:
        st.write("Could not generate cumulative returns plot.")

    st.subheader("📋 Detailed Strategy Results")
    with st.expander("View Detailed Data Table"):
        display_cols = ["company", "date", "sentiment", "signal", "Close", "next_close", "return", "strategy_return", "cumulative_return"]
        valid_cols = [col for col in display_cols if col in results_df_display.columns]
        st.dataframe(results_df_display[valid_cols].reset_index(drop=True))

    st.subheader("💹 Performance Summary")
    summary_data = []
    # Ensure companies_in_results is derived from the actual results, not necessarily companies_to_process
    # as some might have failed during price fetching.
    companies_in_results = results_df_display["company"].unique()
    for company_summary in companies_in_results:
        company_data_summary = results_df_display[results_df_display["company"] == company_summary]
        if not company_data_summary.empty and len(company_data_summary) >= 2:
            final_cr = company_data_summary["cumulative_return"].iloc[-2] - 1
            first_price = company_data_summary["Close"].iloc[0]
            last_price_for_bh = company_data_summary["Close"].iloc[-2] # Align with last strategy day
            bh_return = ((last_price_for_bh / first_price) - 1) if first_price and not pd.isna(first_price) and not pd.isna(last_price_for_bh) else 0

            summary_data.append({
                "Company": company_summary,
                "Final Strategy Return": f"{final_cr:.2%}",
                "Buy & Hold Return": f"{bh_return:.2%}",
                "Strategy vs B&H": f"{final_cr - bh_return:.2%}"
            })
        elif not company_data_summary.empty:
             summary_data.append({
                "Company": company_summary,
                "Final Strategy Return": "N/A (Insufficient data points for full calculation)",
                "Buy & Hold Return": "N/A",
                "Strategy vs B&H": "N/A"
            })
    if summary_data:
        st.table(pd.DataFrame(summary_data))
    else:
        st.write("No summary data to display for the completed strategy.")
else:
    # This 'else' corresponds to st.session_state.strategy_results.empty
    st.info("No strategy results to display. Please run the strategy from the sidebar.")


# --- Optional: Display Raw News Data Used for the last run ---
st.header("📰 Raw News Data Used for Last Strategy Run")
if not st.session_state.news_data.empty: # Display what was used by the strategy
    with st.expander("View Raw News Articles"):
        st.dataframe(st.session_state.news_data)
else:
    st.info("No news data was processed in the last strategy run to display here.")

# --- Optional: Sentiment Visualization ---
# This should also use st.session_state.news_data if it's meant to reflect the last run
st.header("🧐 Sentiment Visualizations (based on last strategy run's news)")
if not st.session_state.news_data.empty:
    df_news_viz = st.session_state.news_data # Already filtered by companies_to_process from the strategy run

    st.subheader("Average Sentiment per Company")
    if not df_news_viz.empty:
        avg_sentiment = df_news_viz.groupby('company')['sentiment'].mean().sort_values()
        st.bar_chart(avg_sentiment)
    else:
        st.write("No data for average sentiment plot.")

    st.subheader("Sentiment Trend Over Time")
    # Use unique companies present in the df_news_viz for selection
    companies_for_trend_plot = df_news_viz['company'].unique().tolist()
    if companies_for_trend_plot:
        company_for_trend = st.selectbox("Select Company for Trend Plot", options=companies_for_trend_plot, key="trend_company_select")
        if company_for_trend:
            df_company_trend = df_news_viz[df_news_viz['company'] == company_for_trend].copy()
            if not df_company_trend.empty and "date" in df_company_trend.columns: # Ensure 'date' exists
                # Group by the 'date' column (which is already date objects)
                daily_sentiment = df_company_trend.groupby("date")["sentiment"].mean()
                st.line_chart(daily_sentiment)
            elif "date" not in df_company_trend.columns:
                st.warning(f"No 'date' column in news data for {company_for_trend} to plot trend.")
            else:
                st.write(f"No news data for {company_for_trend} to plot trend.")
    else:
        st.write("No companies available in the news data for trend plotting.")
else:
    st.info("No news data from the last strategy run to visualize.")

st.sidebar.markdown("---")
st.sidebar.info("Ensure your NewsAPI key is in `config.py`.")