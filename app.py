from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Features list
features = ['ID', 'Age', 'DistanceFromHome', 'MonthlyIncome',
            'JobSatisfaction', 'YearsAtCompany', 'WorkLifeBalance',
            'Education', 'Overtime']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [
            int(request.form['ID']),
            int(request.form['Age']),
            int(request.form['DistanceFromHome']),
            float(request.form['MonthlyIncome']),
            int(request.form['JobSatisfaction']),
            int(request.form['YearsAtCompany']),
            int(request.form['WorkLifeBalance']),
            int(request.form['Education']),
            int(request.form['Overtime'])
        ]

        # Create DataFrame for prediction
        df_input = pd.DataFrame([input_data], columns=features)
        
        # Predict
        prediction = model.predict(df_input)[0]
        result_text = "Employee is likely to leave the company." if prediction == 1 else "Employee is likely to stay."

        return render_template('result.html', result=result_text)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
