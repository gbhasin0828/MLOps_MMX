from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd  # Import pandas for DataFrame operations
import joblib

app = Flask(__name__)

# Load model parameters
model_params = joblib.load('marketing_mix_model.pkl')

# Define hardcoded control variables
HARD_CODED_CONTROLS = [0, 0, 0]  # seas_week_45, seas_week_46, seas_week_47

@app.route('/', methods=['GET'])
def home():
    """
    Render the HTML file for user input.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    try:
        # Parse incoming JSON data
        data = request.json

        if 'media' not in data:
            return jsonify({'error': 'Media input is missing'}), 400

        # Create DataFrame for media inputs
        media_input = pd.DataFrame([data['media']], columns=model_params['media_cols'])

        # Hardcoded control variables as a DataFrame
        control_input = pd.DataFrame([HARD_CODED_CONTROLS], columns=['seas_week_45', 'seas_week_46', 'seas_week_47'])

        # Scale media inputs
        media_scaled = model_params['media_scaler'].transform(media_input)

        # Calculate media effects
        media_effect = 0  # Initialize media_effect
        for i in range(len(model_params['media_cols'])):
            effect = model_params['beta_media'][i] * (
                (media_scaled[:, i] ** model_params['slope'][i]) /
                (model_params['ec'][i] ** model_params['slope'][i] + media_scaled[:, i] ** model_params['slope'][i])
            )
            media_effect += effect  # Accumulate the effect

        # Calculate control effects
        control_effect = np.dot(control_input, model_params['control_coefficients'])

        # Final prediction
        prediction = np.exp(model_params['intercept'] + media_effect + control_effect) - 1

        return jsonify({'prediction': round(prediction[0], 2)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
