import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = None

def load_model():
    global model
    model_path = os.path.join('/opt/ml/model', 'model.pkl')
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")

@app.route('/ping', methods=['GET'])
def ping():
    health = model is not None
    return ('', 200) if health else ('', 404)

@app.route('/invocations', methods=['POST'])
def invoke():
    data = request.get_json()
    inputs = data['inputs']
    predictions = model.predict(inputs)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080)
