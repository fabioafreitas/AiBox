from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, escape, request, json, jsonify
import stock_prediction as stocks
import numpy as np

app = Flask(__name__)

@app.route('/treinar')
def treinar_modelo_endpoint():
    file = "clean_data_index.xlsx"
    index = "csi300 index data"
    mse, rmse, mape, mpe, mae = stocks.evaluate_model_performance(file, index)
    return jsonify({
        'mse':mse,
        'rmse':rmse,
        'mape':mape,
        'mpe':mpe,
        'mae':mae
    })

@app.route('/testar')
def previsao_endpoint():
    previous_data, predicted_data, meta_data = stocks.use_model()
    previous_data = np.reshape(previous_data, (len(previous_data)), order='F')
    predicted_data = np.reshape(predicted_data, (len(predicted_data)), order='F')
    return jsonify({
        'previous_data':previous_data.tolist(),
        'current_data':predicted_data.tolist(),
        'meta_data':meta_data
    })

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8080")