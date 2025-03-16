from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Uih1px5V5tb93au696tC"  # Replace with your Roboflow API key
)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily
    file_path = "temp.jpg"
    file.save(file_path)

    try:
        # Perform inference using the Roboflow API
        result = CLIENT.infer(file_path, model_id="waste-classification-uwqfy/1")
        print("Roboflow API Response:", result)  # Debug print

        # Return the prediction result
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))  # Debug print
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)