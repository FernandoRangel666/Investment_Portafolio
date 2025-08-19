import pandas as pd
import json
import os


def get_sp500_tickers():
    print("Fetching S&P 500 tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]

    # Save to file
    with open("./data/tickers.json", "w") as f:
        json.dump(tickers, f)
    print(f"✅ Saved {len(tickers)} tickers to data/tickers.json")


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    get_sp500_tickers()