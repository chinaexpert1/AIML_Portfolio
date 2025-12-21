# Andrew Taylor
# atayl136

# EN.705.603.81 Creating AI Enabled Systems

from flask import Flask, request, jsonify
import json
from modules.model import Model

app = Flask(__name__)

# Load the model
model = Model(model_type="random")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a transaction is fraudulent.
    Expects a JSON input with transaction details.
    """
    try:
        # Parse JSON request
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Get prediction
        prediction = model.predict(input_data)

        # Return response
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
