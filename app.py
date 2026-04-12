from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)   # ✅ NOT app

model = joblib.load('sentiment_model.joblib')

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': str(prediction)
    })
