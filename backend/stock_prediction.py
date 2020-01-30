import os
from time import sleep

from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Input
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


def split_dataset_train_test(dataset, train_size, test_size):
    begin_of_split = 0
    train_list = []
    test_list = []
    while begin_of_split + train_size + test_size <= dataset.shape[0]:
        train = dataset[begin_of_split:begin_of_split + train_size]
        test = dataset[begin_of_split + train_size:begin_of_split + train_size + test_size]

        begin_of_split += test_size

        train_list.append(train)
        test_list.append(test)

    return train_list, test_list


def split_dataset_train_test_simple(dataset, train_proportion):
    train_size = int(dataset.shape[0] * train_proportion)

    data_train = dataset[:train_size]
    data_test = dataset[train_size:]

    return data_train, data_test


def split_dataset_x_y(dataset):
    x_data = dataset.drop(columns='close')
    y_data = pd.DataFrame(dataset['close'])

    return x_data, y_data


# https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# https://www.dataquest.io/blog/understanding-regression-error-metrics/
def mean_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) / y_true) * 100


#Recebe o caminho até o arquivo e o index do dataframe a ser lido
def load_dataset(file_path, index_name):
    # Carregar de um arquivo excel e mapear cada planilha para um ordered dict
    raw_xlsx_file = pd.ExcelFile(file_path)
    dict_dataframes_index = pd.read_excel(raw_xlsx_file, sheet_name=None)

    df = dict_dataframes_index[index_name]
    # Converter a coluna date para um objeto datetime para plot do gráfico
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    # Definir a coluna data como índice do dataframe
    df = df.set_index('date')

    # Normalização dos dados
    index_backup = df.index.copy()
    columns_names = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    df = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns_names)
    df.index = index_backup
    return df


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

    interval = 'daily'
    series_type = 'open'

    # Fetch dataset from api
    # OHLC series #1
    time_series, meta_time_series = ts_1.get_daily(symbol=symbol, outputsize='full')
    # Bollinger Bands #2
    bbands, meta = ti_1.get_bbands(symbol=symbol, interval=interval, time_period=60, series_type=series_type)
    bbands = bbands.sort_index(ascending=False)
    print(bbands)
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
    print('waiting 70 seconds'); sleep(70)
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
    print('waiting 70 seconds'); sleep(70)
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
    print('waiting 70 seconds'); sleep(70)
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
    print('waiting 70 seconds'); sleep(70)
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

    return merged, meta_time_series


def wavelet_transform(data): #wavelet_reconst_l2
    coeffs = pywt.wavedec(data, 'haar', level=2)
    reconstructed_signal = pywt.waverec(coeffs, 'haar', mode='per')
    return reconstructed_signal


def train_stacked_autoencoder(dataset):
    # Parâmetros das camadas
    input_dim = dataset.shape[1]
    hidden_dim = 10
    activation_function = 'sigmoid'
    train_epochs = 100
    
    # Modelo
    sae = Sequential()
    sae.add(Dense(hidden_dim, activation=activation_function, input_shape=(input_dim,)))
    sae.add(Dense(input_dim, activation=activation_function))
    sae.compile(optimizer='adam', loss='mse')
    sae.fit(dataset, dataset, epochs=train_epochs, verbose=0)

    for i in range(3):
        sae.pop()
        previous_layer = sae.get_layer(index=-1)
        previous_layer.trainable = False
        sae.add(Dense(hidden_dim, activation=activation_function))
        sae.add(Dense(input_dim, activation=activation_function))
        sae.compile(optimizer='adam', loss='mse')
        sae.fit(dataset, dataset, epochs=train_epochs, verbose=0)
        
    return sae


def train_lstm(x_data_train, y_data_train):
    # Parâmetros das camadas
    neurons_lstm = 50
    epochs_number = 40
    # Modelo
    predictor = Sequential()
    predictor.add(LSTM(neurons_lstm, input_shape=(x_data_train.shape[1], x_data_train.shape[2])))
    predictor.add(Dense(1))
    predictor.compile(optimizer='adam', loss='mse')
    predictor.fit(x_data_train, y_data_train, epochs=epochs_number, verbose=0)
    return predictor


def train_model(file_path, index_name, symbol='PETR4.SAO'):
    # Carregando dados do database
    # df = load_dataset(file_path, index_name) # replaced by data from service
    token_hard_coded = ''
    df, meta = get_daily_history(token_1=token_hard_coded, symbol=symbol, amount_of_entries=3500) # Get data from service

    # WAVELET TRANSFORMATION na coluna "close"
    df['close'] = wavelet_transform(df['close'])

    # STACKED AUTOENCODER
    sae = train_stacked_autoencoder(df.drop(columns=['close']))

    # Processando a coluna 'close' no SAE para o LSTM
    close_column_backup = df['close'].copy().values
    df = df.drop(columns=['close'])
    column_backup = df.columns.copy()
    df = sae.predict(df)
    df = pd.DataFrame(df)
    df.columns = column_backup
    df['close'] = close_column_backup

    # Split treino e teste do dataset
    train_data, test_data = split_dataset_train_test_simple(df, train_proportion=0.8)
    x_data_train, y_data_train = split_dataset_x_y(train_data)
    x_data_train = np.reshape(x_data_train.values, (x_data_train.shape[0], x_data_train.shape[1], 1))
    x_data_test, y_data_test = split_dataset_x_y(test_data)
    x_data_test = np.reshape(x_data_test.values, (x_data_test.shape[0], x_data_test.shape[1], 1))

    # LSTM MODELLING AND TRAINING
    lstm = train_lstm(x_data_train, y_data_train)
    lstm.save('model_lstm.h5')
    sae.save('model_sae.h5')

    return x_data_test, y_data_test


def load_trained_model():
    sae = load_model("model_sae.h5")
    lstm = load_model("model_lstm.h5")
    return sae, lstm


# Avalia o desempenho do modelo
# Realiza o treino e teste na proporção 80:20
# Plota o gráfico de comparação do modelo e imprime as métricas
def evaluate_model_performance(file_path, index_name):
    x_data_test, y_data_test = train_model(file_path, index_name)
    sae, lstm = load_trained_model()

    predicted_value = lstm.predict(x_data_test)

    # Fazer um plot do resultado
    plt.figure(figsize=(12, 9))
    plt.title('Conjunto de teste')
    plt.plot(y_data_test.values, 'blue', label='Original')
    plt.plot(predicted_value, 'green', label='LSTM')
    plt.legend()
    plt.savefig('lstm_vs_original_test_set')
    plt.show()

    # Calcular as métricas
    lstm_mse = mean_squared_error(y_data_test.values, predicted_value)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mape = mean_absolute_percentage_error(y_data_test.values, predicted_value)
    lstm_mpe = mean_percentage_error(y_data_test.values, predicted_value)
    lstm_mae = mean_absolute_error(y_data_test.values, predicted_value)

    print(40 * '=')
    print('model performance:')
    print('mse:', lstm_mse, 'rmse:', lstm_rmse, 'mape:', lstm_mape, 'mpe:', lstm_mpe, 'mae:', lstm_mae)

    f = open('predicted_value.txt', 'w')
    f.write(str(predicted_value))
    f.close()

    f = open('original_value.txt', 'w')
    f.write(str(y_data_test.values))
    f.close()

    f = open('metrics_result.txt', 'w')
    f.write('mse: ' + str(lstm_mse) + ' rmse: ' + str(lstm_rmse) + ' mape: ' + str(lstm_mape) + ' mpe: ' + str(
        lstm_mpe) + ' mae: ' + str(lstm_mae))
    f.close()

    return lstm_mse, lstm_rmse, lstm_mape, lstm_mpe, lstm_mae

# Recebe a série temporal do dia atual e tenta prever o valor de fechamento
# Retorna os dados de "fechamento anteriores" e a "previsão"
# TODO Esta função ainda está em desenvolvimento
def use_model():
    sae, lstm = load_trained_model()

    # Load today's data
    token_hard_coded = ''
    symbol_hard_coded = 'PETR4.SAO'
    df, meta_data = get_intraday_data(token_1=token_hard_coded, symbol=symbol_hard_coded, interval='5min')
    df = df[0] # Get most recent entry
    # Apply Wavelet Transformation
    df['close'] = wavelet_transform(df['close'])
    # Pass dataset through Stacked Autoencoder
    close_column_backup = df['close'].copy().values
    df = df.drop(columns=['close'])
    column_backup = df.columns.copy()
    df = sae.predict(df)
    df = pd.DataFrame(df)
    df.columns = column_backup
    df['close'] = close_column_backup
    # Split dataset
    train_data, test_data = split_dataset_train_test_simple(df, train_proportion=0.8)
    x_data_test, y_data_test = split_dataset_x_y(test_data)
    x_data_test = np.reshape(x_data_test.values, (x_data_test.shape[0], x_data_test.shape[1], 1))
    # Predict value using LSTM
    predicted_value = lstm.predict(x_data_test)
    return y_data_test.values, predicted_value, meta_data