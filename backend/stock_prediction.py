from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Input
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt

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

def train_model(file_path, index_name):
    #Carregando dados do database
    df = load_dataset(file_path, index_name)

    # WAVELET TRANSFORMATION na coluna "close"
    df['close'] = wavelet_transform(df['close'])

    # STACKED AUTOENCODER
    sae = train_stacked_autoencoder(df.drop(columns=['close']))

    #Processesando a coluna 'close' no SAE para o LSTM
    close_column_backup = df['close'].copy().values
    df = df.drop(columns=['close'])
    column_backup = df.columns.copy()
    df = sae.predict(df)
    df = pd.DataFrame(df)
    df.columns = column_backup
    df['close'] = close_column_backup

    # Split treino e teste do dataset, observando a restrição de séries temporais
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

#Avalia o desenpenho do modelo
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

#Recebe a série temporal do dia atual e tenta prever o valor de fechamento
#Retorna os dados de "fechamento anteriores" e a "previsão"
#TODO Esta função ainda está em desenvolvimento
def use_model():
    sae, lstm = load_trained_model()

    df = load_dataset("clean_data_index.xlsx", "csi300 index data")
    df['close'] = wavelet_transform(df['close'])
    close_column_backup = df['close'].copy().values
    df = df.drop(columns=['close'])
    column_backup = df.columns.copy()
    df = sae.predict(df)
    df = pd.DataFrame(df)
    df.columns = column_backup
    df['close'] = close_column_backup
    train_data, test_data = split_dataset_train_test_simple(df, train_proportion=0.8)
    x_data_train, y_data_train = split_dataset_x_y(train_data)
    x_data_train = np.reshape(x_data_train.values, (x_data_train.shape[0], x_data_train.shape[1], 1))
    x_data_test, y_data_test = split_dataset_x_y(test_data)
    x_data_test = np.reshape(x_data_test.values, (x_data_test.shape[0], x_data_test.shape[1], 1))

    predicted_value = lstm.predict(x_data_test)
    return y_data_test.values, predicted_value