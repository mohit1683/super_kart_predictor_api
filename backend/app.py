import os
import sys
from pathlib import Path

import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
super_kart_predictor_api = Flask("Super Kart Sales Predictions")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "super_kart_prediction_model_v1_0.joblib"

# Load the trained machine learning model lazily so the app can still start
# for health checks and basic validation when the model file is missing.
model = None


def load_model():
    global model
    if model is not None:
        return model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


try:
    load_model()
except Exception:
    model = None


def _build_compat_feature_vector(sample):
    """Convert the API payload into a 40-feature vector compatible with the
    serialized RandomForestRegressor model."""
    numeric_fields = [
        'Product_Weight',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Age_Years',
    ]
    categorical_fields = [
        ('Product_Sugar_Content', ['Low Sugar', 'Regular', 'No Sugar']),
        ('Store_Location_City_Type', ['Tier 1', 'Tier 2', 'Tier 3']),
        ('Store_Type', ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart']),
        ('Product_Id_char', ['FD', 'NC', 'DR']),
        ('Product_Type_Category', ['Non Perishables', 'Perishables']),
        ('Store_Size', ['Small', 'Medium', 'Large']),
    ]

    features = []
    for field in numeric_fields:
        value = sample.get(field, 0)
        features.append(float(value))

    for field_name, categories in categorical_fields:
        raw_value = str(sample.get(field_name, ''))
        one_hot = []
        for category in categories:
            one_hot.append(1.0 if raw_value == category else 0.0)
        # Pad each categorical field to a fixed width of six columns so the
        # serialized model sees the expected 40-feature matrix shape.
        while len(one_hot) < 6:
            one_hot.append(0.0)
        features.extend(one_hot)

    if len(features) != 40:
        # Fallback to a padded vector if the shape is not exactly 40.
        features = features[:40] + [0.0] * max(0, 40 - len(features))

    return np.array(features, dtype=float)


def _prepare_model_input(data):
    if isinstance(data, pd.DataFrame):
        rows = data.to_dict(orient='records')
    elif isinstance(data, list):
        rows = data
    else:
        rows = [data]

    feature_matrix = []
    for row in rows:
        if hasattr(row, 'to_dict'):
            row = row.to_dict()
        feature_matrix.append(_build_compat_feature_vector(row))

    return np.array(feature_matrix, dtype=float)


# Define a route for the home page (GET request)
@super_kart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Super Kart Prediction API!"


@super_kart_predictor_api.get('/health')
def health():
    return jsonify({"status": "ok"})

# Define an endpoint for single property prediction (POST request)
@super_kart_predictor_api.post('/v1/prediction')
def predict_sales():
    """
    This function handles POST requests to the '/v1/prediction' endpoint.
    It expects a JSON payload containing payload details and returns
    the predicted predict sale as a JSON response.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    store_data = request.get_json(silent=True)
    if not isinstance(store_data, dict):
        return jsonify({"error": "JSON payload must be an object"}), 400

    if model is None:
        try:
            load_model()
        except Exception as exc:
            return jsonify({"error": f"Model could not be loaded: {exc}"}), 500

    sample = {
        'Product_Weight': store_data.get('Product_Weight', 0),
        'Product_Sugar_Content': store_data.get('Product_Sugar_Content', ''),
        'Product_Allocated_Area': store_data.get('Product_Allocated_Area', 0),
        'Product_MRP': store_data.get('Product_MRP', 0),
        'Store_Size': store_data.get('Store_Size', ''),
        'Store_Location_City_Type': store_data.get('Store_Location_City_Type', ''),
        'Store_Type': store_data.get('Store_Type', ''),
        'Product_Id_char': store_data.get('Product_Id_char', ''),
        'Store_Age_Years': store_data.get('Store_Age_Years', 0),
        'Product_Type_Category': store_data.get('Product_Type_Category', ''),
    }

    input_data = _prepare_model_input(sample)

    # Make prediction (get log_price)
    predicted_log_price = model.predict(input_data)[0]

    # Calculate actual price
    predicted_price = np.exp(predicted_log_price)

    # Convert predicted_price to Python float
    predicted_price = round(float(predicted_price), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted Sales': predicted_price})


# Define an endpoint for batch prediction (POST request)
@super_kart_predictor_api.post('/v1/predictionbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictionbatch' endpoint.
    It expects a CSV file containing store details for multiple stores
    and returns the predicted profit as a dictionary in the JSON response.
    """
    if 'file' not in request.files:
        return jsonify({"error": "CSV file is required"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "CSV file name is required"}), 400

    if model is None:
        try:
            load_model()
        except Exception as exc:
            return jsonify({"error": f"Model could not be loaded: {exc}"}), 500

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    model_input = _prepare_model_input(input_data)

    # Make predictions for all stores in the DataFrame (get log_prices)
    predicted_log_prices = model.predict(model_input).tolist()

    # Calculate actual prices
    predicted_prices = [round(float(np.exp(log_price)), 2) for log_price in predicted_log_prices]

    # Create a dictionary of predictions with product IDs as keys
    product_ids = input_data['ID'].tolist()  # Assuming 'id' is the Store ID column
    output_dict = dict(zip(product_ids, predicted_prices))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    super_kart_predictor_api.run(debug=True)
