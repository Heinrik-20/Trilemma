import yfinance as yf

from datetime import datetime


def extract_btc():
    today = datetime.today().strftime("%Y-%m-%d")
    btc = yf.Ticker('BTC-USD').history(period="730d", interval='1h').drop(columns=['Dividends', 'Stock Splits'])
    btc.to_parquet(f"../data/btc-{today}.pq")

    return


if __name__ == "__main__":
    extract_btc()