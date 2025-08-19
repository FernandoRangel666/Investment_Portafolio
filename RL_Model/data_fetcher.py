# scripts/data_fetcher.py
import yfinance as yf
import pandas as pd
import json
import os

def load_tickers():
    with open("./data/tickers.json") as f:
        return json.load(f)

def fetch_data():

    # Optional: pick top 20 large caps (or use all)
    selected_tickers = load_tickers()

    print(f"Downloading data for {len(selected_tickers)} stocks...")
    start_date = "2020-08-01"
    end_date = "2025-08-01"
    data = yf.download(selected_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Extract Close prices
    close_data = pd.concat(
        [data[ticker]['Close'] for ticker in selected_tickers],
        axis=1,
        keys=selected_tickers  # preserves the ticker names as column names
    )
    close_data.dropna(inplace=True)

    close_data.to_pickle("./data/stock_data.pkl")
    print(f"✅ Saved price data to data/stock_data.pkl (shape: {close_data.shape})")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    fetch_data()