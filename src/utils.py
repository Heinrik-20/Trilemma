import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

from datetime import datetime
from ta.trend import ema_indicator, stc, macd, trix, kst
from ta.volatility import bollinger_hband, bollinger_lband, ulcer_index
from ta.momentum import kama, ppo, roc, rsi, stochrsi, tsi


# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BTC = "BTC-USD"
END_DATE = datetime.today().strftime("%Y-%m-%d")

def create_dataset():

    stocks = yf.Ticker(
        BTC
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        btc = stocks.history(start='2022-09-12', end=END_DATE, interval='1h').drop(columns=['Dividends', 'Stock Splits']).reset_index()
        btc = btc.rename(columns={'Datetime': 'Date'})
        btc_old = pd.read_parquet("../data/btc-2015-2023.pq").drop(columns=['Volume USD'])
        btc_old['Date'] = pd.to_datetime(btc_old['Date']).dt.tz_localize('UTC')
        btc_old = btc_old.loc[btc_old['Date'] < pd.to_datetime("2022-09-12 00:00:00+00:00")]

        btc_2019 = pd.read_parquet("../data/btc-2019.pq")
        btc_2019 = btc_2019.rename(columns={
            col: col.capitalize() for col in btc_2019.columns
        })
        btc_2019['Date'] = pd.to_datetime(btc_2019['Date']).dt.tz_localize('UTC')

        btc_2018 = pd.read_parquet("../data/btc-2018.pq")
        btc_2018 = btc_2018.rename(columns={
            col: col.capitalize() for col in btc_2018.columns
        })
        btc_2018['Date'] = pd.to_datetime(btc_2018['Date']).dt.tz_localize('UTC')


        btc_old = pd.concat([btc_old, btc_2018, btc_2019])

        def get_open(x: pd.Series):
            return x[x.index == np.min(x.index)]

        mapper = {
            'Open': get_open,
        }
        btc_old_processed = btc_old.sort_values(by='Date').reset_index(drop=True).groupby(pd.Grouper(key='Date', freq='H')).agg(mapper).reset_index()
        
        btc = pd.concat([btc, btc_old_processed], axis=0).sort_values(by='Date').reset_index(drop=True)[['Date', 'Open']]
        btc = btc.loc[(btc['Date'].dt.day_of_week == 0) & (btc['Date'].dt.hour == 20)]

    # Trend indicators
    btc['ema_12'] = ema_indicator(btc['Open'])
    btc['ema_26'] = ema_indicator(btc['Open'], window=26)
    btc['stc'] = stc(btc['Open'])
    btc['macd'] = macd(btc['Open'])
    btc['trix'] = trix(btc['Open'])
    btc['kst'] = kst(btc['Open'])

    # Volatility indicators
    btc['bollinger_high'] = bollinger_hband(btc['Open'])
    btc['bollinger_low'] = bollinger_lband(btc['Open'])
    btc['ulcer_index'] = ulcer_index(btc['Open'])

    # Momentum indicators
    btc['kama'] = kama(btc['Open'])
    btc['ppo'] = ppo(btc['Open'])
    btc['roc'] = roc(btc['Open'])
    btc['rsi'] = rsi(btc['Open'])
    btc['stochrsi'] = stochrsi(btc['Open'])
    btc['tsi'] = tsi(btc['Open'])

    btc['target'] = ((btc['Open'].shift(-1).fillna(np.nan).values - btc['Open'].values)/btc['Open'].values) * 100
    btc = btc.dropna()
    
    return btc
