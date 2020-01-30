from flask import Flask, escape, request, json, jsonify
import stock_prediction as stocks
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/train')
def treinar_modelo_endpoint():
    symbol = request.args.get('symbol')
    if symbol != "PETR4.SAO" and symbol != "VALE3.SAO" and symbol != "ITUB4.SAO":
        return jsonify("Só são aceitos no momento as ações da: Pretrobras (PETR4.SAO), Vale (VALE3.SAO) e Itau (ITUB4.SAO)"), 404

    mse, rmse, mape, mpe, mae = stocks.evaluate_model_performance(symbol=symbol)
    return jsonify({
        'mse':mse,
        'rmse':rmse,
        'mape':mape,
        'mpe':mpe,
        'mae':mae
    })

@app.route('/predict')
def previsao_endpoint():
    symbol = request.args.get('symbol')
    if symbol != "PETR4.SAO" and symbol != "VALE3.SAO" and symbol != "ITUB4.SAO":
        return jsonify("Só são aceitos no momento as ações da: Pretrobras (PETR4.SAO), Vale (VALE3.SAO) e Itau (ITUB4.SAO)"), 404

    previous_data, predicted_data = stocks.use_model(symbol)
    previous_data = np.reshape(previous_data, (len(previous_data)), order='F')
    predicted_data = np.reshape(predicted_data, (len(predicted_data)), order='F')
    return jsonify({
        'previous_data':previous_data.tolist(),
        'current_data':predicted_data.tolist()
    })

@app.route('/history')
def get_history_endpoint():
    symbol = request.args.get('symbol')
    if symbol != "PETR4.SAO" and symbol != "VALE3.SAO" and symbol != "ITUB4.SAO":
        return jsonify("Só são aceitos no momento as ações da: Pretrobras (PETR4.SAO), Vale (VALE3.SAO) e Itau (ITUB4.SAO)"), 404
    

    df = pd.read_csv('dataset.csv', index_col=['date'])[1:]
    return jsonify({
        'date': df.index.tolist(),
        'close': df['close'].values.tolist(),
    })


@app.route('/')
def teste():
    return jsonify("Servidor Online!")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="3000")