import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the preprocessed data and model
model_path = "model.pkl"  # Save your trained model as 'model.pkl'


# Load trained model
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return "Welcome to the Poverty Prediction API!"

@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()  # Retrieve JSON data
    print(data) 

# Define a route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get JSON data from the request
        data = request.get_json()  # Expecting the data in JSON format

        # Step 2: Convert the JSON data into a pandas DataFrame
        input_data = pd.DataFrame([data])

        # Step 3: Make predictions using the model
        prediction = model.predict(input_data)

        # Step 4: Return the prediction as a JSON response
        # In case of regression, the prediction is a numeric value (you can adapt the response to your needs)
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # If an error occurs, send an error message in the response
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
    
