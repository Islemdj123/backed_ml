from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

model_path = 'models/knn_model.pkl'
scaler_path = 'models/scaler.pkl'
label_encoder_path = 'models/label_encoder.pkl'

knn_model = None
scaler = None
label_encoder = None

def load_models():
    global knn_model, scaler, label_encoder
    try:
        knn_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        print("Models loaded successfully!")
    except FileNotFoundError:
        print("ERROR: Model files not found. Please run train_model.py first.")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Smart Hospital ML Backend is running'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Support both flat and nested structure for compatibility
        lab_results = data.get('lab_results', {})
        
        # Mapping incoming data to the 17 features required by the model
        # Using typical normal values as defaults if not provided
        features = [
            lab_results.get('hemoglobin') or data.get('hemoglobin') or 14.0,
            lab_results.get('rbc') or data.get('red_blood_cells') or 5.0,
            lab_results.get('wbc') or data.get('white_blood_cells') or 7.0,
            lab_results.get('platelets') or data.get('platelets') or 250.0,
            lab_results.get('mcv') or data.get('mean_corpuscular_volume') or 90.0,
            lab_results.get('glucose') or data.get('glucose') or 100.0,
            lab_results.get('cholesterol') or data.get('cholesterol') or 180.0,
            lab_results.get('triglycerides') or data.get('triglycerides') or 150.0,
            lab_results.get('creatinine') or data.get('creatinine') or 1.0,
            lab_results.get('urea') or data.get('urea') or 30.0,
            lab_results.get('crp') or data.get('c_reactive_protein') or 0.5,
            data.get('blood_pressure') or 120.0,
            data.get('heart_rate') or 75.0,
            data.get('bmi') or 24.0,
            lab_results.get('sodium') or data.get('sodium') or 140.0,
            lab_results.get('potassium') or data.get('potassium') or 4.0,
            lab_results.get('calcium') or data.get('calcium') or 9.5
        ]
        
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        prediction = knn_model.predict(features_scaled)[0]
        probabilities = knn_model.predict_proba(features_scaled)[0]
        
        diagnosis = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities)) * 100
        
        response_data = {
            'diagnosis': str(diagnosis),
            'confidence': round(confidence, 2),
            'model': 'KNN_v1.0',
            'status': 'success',
            'note': 'This is a preliminary diagnosis. Always consult with a medical professional.'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'application': 'Smart Hospital - Preliminary Diagnosis System',
        'version': '1.0.0',
        'ml_algorithm': 'K-Nearest Neighbors (KNN)',
        'description': 'ML-powered decision support system for preliminary medical diagnosis',
        'disclaimer': 'This system provides PRELIMINARY diagnosis only and does NOT replace professional medical advice.',
        'supported_diagnoses': list(label_encoder.classes_)
    }), 200

# Load models at startup
try:
    load_models()
except Exception as e:
    print(f"Warning during startup: {e}")

if __name__ == '__main__':
    # Use PORT environment variable for Render compatibility
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
