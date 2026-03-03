import pandas as pd



def get_sp500_symbols():
    """Scrapes the current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    tickers = table["Symbol"].tolist()
    # Clean up tickers for Alpaca compatibility (e.g., BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers
