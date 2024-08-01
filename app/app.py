# app/app.py
from flask import Flask, request, jsonify
import pickle

# Load the model
model = pickle.load(open('model/credit_risk_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
