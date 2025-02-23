import joblib
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model_pipeline.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    X = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(X)
    print(output[0])
    return jsonify({'prediction': int(str(output[0]))}), 200


if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    # feature_0	feature_1	feature_2	feature_3	feature_4

