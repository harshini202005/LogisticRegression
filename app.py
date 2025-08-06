from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [
        float(request.form['ID']),
        float(request.form['Age']),
        float(request.form['Experience']),
        float(request.form['Income']),
        float(request.form['CCAvg']),
        float(request.form['Education']),
        float(request.form['Mortgage']),
        float(request.form['CD_Account']),
        float(request.form['CreditCard'])
    ]
    final_input = np.array([input_features])
    result = model.predict(final_input)[0]
    return render_template('result.html', prediction='Approved' if result == 1 else 'Rejected')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
