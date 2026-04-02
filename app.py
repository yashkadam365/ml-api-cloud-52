from flask import Flask, request, jsonify
import joblib
import os

application = Flask(__name__)

# Load model safely
model = joblib.load('sentiment_model.joblib')

@application.route('/')
def home():
    return "App is running!", 200

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port)
