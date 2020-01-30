from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from time import sleep

import pandas as pd
import os


def fetch_stock_history(filename, token_1, symbol, amount_of_entries=2500):
    # Get entiry history from service
    dataset, meta_data = get_daily_history(token_1=token_1, symbol=symbol, amount_of_entries=amount_of_entries)
    # Store this history in a specific directory
    dataset_dirname = f'dataset_{symbol.lower}'
    os.mkdir(dataset_dirname)
    os.chdir(dataset_dirname)
    file = open(symbol.lower, 'w')
    file.write(dataset_dirname)
    os.chdir('..')


def get_daily_history(token_1, symbol, amount_of_entries=1000):
    ts_1 = TimeSeries(key=token_1, output_format='pandas')
    ti_1 = TechIndicators(key=token_1, output_format='pandas')

    print('get_daily_history')

    interval = 'daily'
    series_type = 'open'

    # Fetch dataset from api
    # OHLC series #1
    time_series, meta_time_series = ts_1.get_daily(symbol=symbol, outputsize='full')
    # Bollinger Bands #2
    bbands, meta = ti_1.get_bbands(symbol=symbol, interval=interval, time_period=60, series_type=series_type)
    bbands = bbands.sort_index(ascending=False)
    # SMA10 or MA10 #3
    sma_10, meta = ti_1.get_sma(symbol=symbol, interval=interval, time_period=10, series_type=series_type)
    sma_10 = sma_10.sort_index(ascending=False)
    # SMA5 or MA5 #4
    sma_5, meta = ti_1.get_sma(symbol=symbol, interval=interval, time_period=5, series_type=series_type)
    sma_5 = sma_5.sort_index(ascending=False)
    # ROC #5
    roc_20, meta = ti_1.get_roc(symbol=symbol, interval=interval, time_period=20, series_type=series_type)
    roc_20 = roc_20.sort_index(ascending=False)
    # Wait 70 seconds
    print('waiting 70 seconds');
    sleep(70)
    # MACD #1
    macd, meta = ti_1.get_macd(symbol=symbol, interval=interval, series_type=series_type)
    macd = macd.sort_index(ascending=False)
    # CCI #2
    cci_20, meta = ti_1.get_cci(symbol=symbol, interval=interval, time_period=20)
    cci_20 = cci_20.sort_index(ascending=False)
    # ATR #3
    atr_20, meta = ti_1.get_atr(symbol=symbol, interval=interval, time_period=20)
    atr_20 = atr_20.sort_index(ascending=False)
    # EMA20 #4
    ema_20, meta = ti_1.get_ema(symbol=symbol, interval=interval, time_period=20, series_type=series_type)
    ema_20 = ema_20.sort_index(ascending=False)
    # MTM6 #5
    mtm_6, meta = ti_1.get_mom(symbol=symbol, interval=interval, time_period=180, series_type=series_type)
    mtm_6 = mtm_6.sort_index(ascending=False)
    # Wait 70 seconds
    print('waiting 70 seconds');
    sleep(70)
    # MTM12 #1
    mtm_12, meta = ti_1.get_mom(symbol=symbol, interval=interval, time_period=360, series_type=series_type)
    mtm_12 = mtm_12.sort_index(ascending=False)

    # Get last n data points in the dataset and set your respective column name
    d1 = time_series[:amount_of_entries];
    d1.columns = ['open', 'high', 'low', 'close', 'volume']
    d2 = sma_10[:amount_of_entries]
    d2.columns = ['sma10']
    d3 = sma_5[:amount_of_entries]
    d3.columns = ['sma5']
    d4 = pd.DataFrame(bbands['Real Middle Band'][:amount_of_entries])
    d4.columns = ['bbands']
    d5 = roc_20[:amount_of_entries]
    d5.columns = ['roc']
    d6 = pd.DataFrame(macd['MACD'][:amount_of_entries])
    d6.columns = ['macd']
    d7 = cci_20[:amount_of_entries]
    d7.columns = ['cci']
    d8 = atr_20[:amount_of_entries]
    d8.columns = ['atr']
    d9 = ema_20[:amount_of_entries]
    d9.columns = ['ema20']
    d10 = mtm_6[:amount_of_entries]
    d10.columns = ['mtm6']
    d11 = mtm_12[:amount_of_entries]
    d11.columns = ['mtm12']
    # Merge elements
    merged = d1.merge(d2, left_index=True, right_index=True)
    merged = merged.merge(d3, left_index=True, right_index=True)
    merged = merged.merge(d4, left_index=True, right_index=True)
    merged = merged.merge(d5, left_index=True, right_index=True)
    merged = merged.merge(d6, left_index=True, right_index=True)
    merged = merged.merge(d7, left_index=True, right_index=True)
    merged = merged.merge(d8, left_index=True, right_index=True)
    merged = merged.merge(d9, left_index=True, right_index=True)
    merged = merged.merge(d10, left_index=True, right_index=True)
    merged = merged.merge(d11, left_index=True, right_index=True)

    sep = os.path.sep
    filename = 'datasets'+sep+'history_'+symbol+'.csv'
    merged.to_csv(filename, index=True)
    return merged, meta_time_series


def get_intraday_data(token_1, symbol, interval='15min'):
    ts_1 = TimeSeries(key=token_1, output_format='pandas')
    ti_1 = TechIndicators(key=token_1, output_format='pandas')
    series_type = 'open'
    # Fetch dataset from api
    # OHLC series #1
    time_series, meta_time_series = ts_1.get_intraday(symbol=symbol, interval=interval, outputsize='compact')
    # Bollinger Bands #2
    bbands, meta = ti_1.get_bbands(symbol=symbol, interval=interval, time_period=60, series_type=series_type)
    bbands = bbands.sort_index(ascending=False)
    # SMA10 or MA10 #3
    sma_10, meta = ti_1.get_sma(symbol=symbol, interval=interval, time_period=10, series_type=series_type)
    sma_10 = sma_10.sort_index(ascending=False)
    # SMA5 or MA5 #4
    sma_5, meta = ti_1.get_sma(symbol=symbol, interval=interval, time_period=5, series_type=series_type)
    sma_5 = sma_5.sort_index(ascending=False)
    # ROC #5
    roc_20, meta = ti_1.get_roc(symbol=symbol, interval=interval, time_period=20, series_type=series_type)
    roc_20 = roc_20.sort_index(ascending=False)
    # Wait 70 seconds
    print('waiting 70 seconds');
    sleep(70)
    # MACD #1
    macd, meta = ti_1.get_macd(symbol=symbol, interval=interval, series_type=series_type)
    macd = macd.sort_index(ascending=False)
    # CCI #2
    cci_20, meta = ti_1.get_cci(symbol=symbol, interval=interval, time_period=20)
    cci_20 = cci_20.sort_index(ascending=False)
    # ATR #3
    atr_20, meta = ti_1.get_atr(symbol=symbol, interval=interval, time_period=20)
    atr_20 = atr_20.sort_index(ascending=False)
    # EMA20 #4
    ema_20, meta = ti_1.get_ema(symbol=symbol, interval=interval, time_period=20, series_type=series_type)
    ema_20 = ema_20.sort_index(ascending=False)
    # MTM6 #5
    mtm_6, meta = ti_1.get_mom(symbol=symbol, interval=interval, time_period=180, series_type=series_type)
    mtm_6 = mtm_6.sort_index(ascending=False)
    # Wait 70 seconds
    print('waiting 70 seconds');
    sleep(70)
    # MTM12 #1
    mtm_12, meta = ti_1.get_mom(symbol=symbol, interval=interval, time_period=360, series_type=series_type)
    mtm_12 = mtm_12.sort_index(ascending=False)

    # Get last n data points in the dataset and set your respective column name
    amount_of_entries = time_series.shape[0]
    d1 = time_series[:amount_of_entries]
    d1.columns = ['open', 'high', 'low', 'close', 'volume']
    d2 = sma_10[:amount_of_entries]
    d2.columns = ['sma10']
    d3 = sma_5[:amount_of_entries]
    d3.columns = ['sma5']
    d4 = pd.DataFrame(bbands['Real Middle Band'][:amount_of_entries])
    d4.columns = ['bbands']
    d5 = roc_20[:amount_of_entries]
    d5.columns = ['roc']
    d6 = pd.DataFrame(macd['MACD'][:amount_of_entries])
    d6.columns = ['macd']
    d7 = cci_20[:amount_of_entries]
    d7.columns = ['cci']
    d8 = atr_20[:amount_of_entries]
    d8.columns = ['atr']
    d9 = ema_20[:amount_of_entries]
    d9.columns = ['ema20']
    d10 = mtm_6[:amount_of_entries]
    d10.columns = ['mtm6']
    d11 = mtm_12[:amount_of_entries]
    d11.columns = ['mtm12']
    # Merge elements
    merged = d1.merge(d2, left_index=True, right_index=True)
    merged = merged.merge(d3, left_index=True, right_index=True)
    merged = merged.merge(d4, left_index=True, right_index=True)
    merged = merged.merge(d5, left_index=True, right_index=True)
    merged = merged.merge(d6, left_index=True, right_index=True)
    merged = merged.merge(d7, left_index=True, right_index=True)
    merged = merged.merge(d8, left_index=True, right_index=True)
    merged = merged.merge(d9, left_index=True, right_index=True)
    merged = merged.merge(d10, left_index=True, right_index=True)
    merged = merged.merge(d11, left_index=True, right_index=True)

    sep = os.path.sep
    filename = 'datasets'+sep+'intraday_'+symbol+'.csv'
    merged.to_csv(filename, index=True)
    return merged, meta_time_series