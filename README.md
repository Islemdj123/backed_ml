# Smart Hospital - Python ML Backend

## 📌 Overview

This is the Machine Learning backend for the Smart Hospital system. It provides a REST API for preliminary medical diagnosis using the K-Nearest Neighbors (KNN) algorithm.

**Technology:** Python 3.8+ | Flask | scikit-learn

---

## 📦 Project Structure

```
backend_ml/
├── app.py                  # Flask REST API server
├── train_model.py         # KNN model training script
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── data/
│   └── medical_dataset.csv # Training dataset
├── models/                # Trained models (created after training)
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── README.md             # This file
```

---

## 🚀 Quick Start

### Data source: CSV or MongoDB
This backend supports training the KNN model from either a CSV file (default) or a MongoDB collection.

- CSV (default):
  - DATA_SOURCE=csv
  - DATA_PATH=data/medical_dataset.csv

- MongoDB (optional):
  - DATA_SOURCE=mongodb
  - MONGO_URI=mongodb://localhost:27017
  - MONGO_DB=smart_hospital
  - MONGO_COLLECTION=medical_dataset
  - MONGO_QUERY (optional JSON string filter)

Example .env for MongoDB:
```
DATA_SOURCE=mongodb
MONGO_URI=mongodb://localhost:27017
MONGO_DB=smart_hospital
MONGO_COLLECTION=medical_dataset
# MONGO_QUERY={"diagnosis": {"$in": ["Normal", "Diabetes", "Hypertension"]}}
```

Model artifact paths can be customized:
- MODEL_PATH=models/knn_model.pkl
- SCALER_PATH=models/scaler.pkl
- LABEL_ENCODER_PATH=models/label_encoder.pkl

### 1. Setup Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

Expected output:
```
Loading medical dataset...
Dataset shape: (30, 11)
Features: ['age', 'gender', 'temperature', ...]
Diagnosis distribution:
 Normal           10
 Hypertension     10
 Diabetes         10

Training set size: 24
Test set size: 6

==================================================
Model Performance Metrics:
==================================================
Accuracy:  0.8333
Precision: 0.8500
Recall:    0.8333
F1-Score:  0.8367
==================================================

Model saved to: models/knn_model.pkl
Scaler saved to: models/scaler.pkl
Label Encoder saved to: models/label_encoder.pkl

Training completed successfully!
```

### 4. Start the API Server

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: off
 * WARNING: This is a development server. Do not use it in production.
```

---

## 🔌 API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Check if the backend is running

**Response:**
```json
{
  "status": "healthy",
  "message": "Smart Hospital ML Backend is running"
}
```

---

### 2. Backend Information

**Endpoint:** `GET /info`

**Purpose:** Get information about the ML system

**Response:**
```json
{
  "application": "Smart Hospital - Preliminary Diagnosis System",
  "version": "1.0.0",
  "ml_algorithm": "K-Nearest Neighbors (KNN)",
  "description": "ML-powered decision support system for preliminary medical diagnosis",
  "disclaimer": "This system provides PRELIMINARY diagnosis only and does NOT replace professional medical advice.",
  "supported_diagnoses": [
    "Normal",
    "Hypertension",
    "Diabetes"
  ]
}
```

---

### 3. Get Preliminary Diagnosis

**Endpoint:** `POST /predict`

**Purpose:** Get preliminary diagnosis based on clinical data

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 35,
  "gender": 1,
  "temperature": 37.2,
  "blood_pressure_systolic": 120,
  "blood_pressure_diastolic": 80,
  "heart_rate": 72,
  "glucose": 95,
  "hemoglobin": 13.5,
  "white_blood_cells": 7.2,
  "platelets": 250
}
```

**Field Descriptions:**
| Field | Type | Range | Required | Description |
|-------|------|-------|----------|-------------|
| age | int | 1-120 | Yes | Patient age in years |
| gender | int | 0-1 | Yes | 0=Male, 1=Female |
| temperature | float | 35.0-42.0 | Yes | Body temperature in °C |
| blood_pressure_systolic | int | 80-240 | Yes | Systolic BP in mmHg |
| blood_pressure_diastolic | int | 40-160 | Yes | Diastolic BP in mmHg |
| heart_rate | int | 30-200 | Yes | Heart rate in bpm |
| glucose | int | 40-500 | Yes | Blood glucose in mg/dL |
| hemoglobin | float | 5.0-20.0 | Yes | Hemoglobin in g/dL |
| white_blood_cells | float | 1.0-30.0 | Yes | WBC in K/uL |
| platelets | int | 10-1000 | Yes | Platelet count in K/uL |

**Successful Response (200):**
```json
{
  "diagnosis": "Normal",
  "confidence": 85.50,
  "note": "This is a preliminary diagnosis. Always consult with a medical professional.",
  "input_data": {
    "age": 35,
    "gender": 1,
    "temperature": 37.2,
    "blood_pressure": "120/80",
    "heart_rate": 72,
    "glucose": 95,
    "hemoglobin": 13.5,
    "white_blood_cells": 7.2,
    "platelets": 250
  }
}
```

**Error Response (400 - Missing Fields):**
```json
{
  "error": "Missing required fields",
  "required_fields": [
    "age", "gender", "temperature", "blood_pressure_systolic",
    "blood_pressure_diastolic", "heart_rate", "glucose",
    "hemoglobin", "white_blood_cells", "platelets"
  ]
}
```

**Error Response (500 - Server Error):**
```json
{
  "error": "error message here",
  "message": "Prediction failed"
}
```

---

## 📊 Machine Learning Model

### Algorithm: K-Nearest Neighbors (KNN)

**Why KNN for Medical Decision Support?**
- ✅ Simple and interpretable
- ✅ No complex hyperparameters to tune
- ✅ Works well with small to medium datasets
- ✅ Easy to explain to healthcare professionals
- ✅ Instance-based learning (transparent)

**Model Specifications:**
- **Algorithm:** K-Nearest Neighbors
- **Distance Metric:** Euclidean distance
- **Number of Neighbors (k):** 5
- **Feature Scaling:** StandardScaler (z-score normalization)
- **Training Set Size:** 24 records (80%)
- **Test Set Size:** 6 records (20%)

**Training Process:**

```python
# 1. Load and prepare data
df = pd.read_csv('data/medical_dataset.csv')

# 2. Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 3. Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# 7. Save model
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
```

---

## 📈 Performance Metrics

When you run `python train_model.py`, you'll see metrics like:

```
Accuracy:  0.8333  (% of correct predictions)
Precision: 0.8500  (accuracy of positive predictions)
Recall:    0.8333  (% of actual positives found)
F1-Score:  0.8367  (harmonic mean of precision and recall)
```

**Interpretation:**
- **Accuracy:** 83.33% - 5 out of 6 test cases correctly classified
- **Precision:** 85% - When model predicts positive, it's correct 85% of the time
- **Recall:** 83.33% - Model finds 83.33% of actual positive cases
- **F1-Score:** 83.67% - Overall model effectiveness

---

## 📚 Training Dataset

The system includes a sample dataset in `data/medical_dataset.csv` with 30 records across 3 diagnostic categories:

**Classes:**
- **Normal:** 10 records
- **Hypertension:** 10 records  
- **Diabetes:** 10 records

**Features:**
- Age (18-60 years)
- Gender (Male=0, Female=1)
- Temperature (36.8-40.0°C)
- Blood Pressure (120/80 - 170/110 mmHg)
- Heart Rate (50-110 bpm)
- Glucose (88-180 mg/dL)
- Hemoglobin (11.3-14.4 g/dL)
- White Blood Cells (6.6-15.0 K/uL)
- Platelets (170-290 K/uL)

---

## 🔧 Configuration

### Environment Variables (.env)

```env
FLASK_ENV=development      # development or production
FLASK_PORT=5000           # Port number (default: 5000)
DEBUG=False               # Enable/disable debug mode
```

For production, also set:

```env
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO
```

---

## 🧪 Testing

### Test with cURL

```bash
# Test 1: Health Check
curl http://localhost:5000/health

# Test 2: Get Backend Info
curl http://localhost:5000/info

# Test 3: Get Diagnosis (Normal Case)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": 1,
    "temperature": 37.2,
    "blood_pressure_systolic": 120,
    "blood_pressure_diastolic": 80,
    "heart_rate": 72,
    "glucose": 95,
    "hemoglobin": 13.5,
    "white_blood_cells": 7.2,
    "platelets": 250
  }'

# Test 4: Get Diagnosis (Hypertension Case)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "gender": 0,
    "temperature": 38.8,
    "blood_pressure_systolic": 155,
    "blood_pressure_diastolic": 100,
    "heart_rate": 92,
    "glucose": 140,
    "hemoglobin": 12.2,
    "white_blood_cells": 10.2,
    "platelets": 190
  }'

# Test 5: Get Diagnosis (Diabetes Case)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": 0,
    "temperature": 39.2,
    "blood_pressure_systolic": 160,
    "blood_pressure_diastolic": 105,
    "heart_rate": 98,
    "glucose": 165,
    "hemoglobin": 11.9,
    "white_blood_cells": 12.5,
    "platelets": 220
  }'
```

### Test with Python

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# Test prediction
payload = {
    "age": 35,
    "gender": 1,
    "temperature": 37.2,
    "blood_pressure_systolic": 120,
    "blood_pressure_diastolic": 80,
    "heart_rate": 72,
    "glucose": 95,
    "hemoglobin": 13.5,
    "white_blood_cells": 7.2,
    "platelets": 250
}

response = requests.post(f"{BASE_URL}/predict", json=payload)
print(response.json())
```

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
Flask==2.3.2              # Web framework
Flask-CORS==4.0.0         # Cross-Origin Resource Sharing
scikit-learn==1.3.0       # ML algorithms
pandas==2.0.3             # Data manipulation
numpy==1.24.3             # Numerical computing
joblib==1.3.1             # Model persistence
python-dotenv==1.0.0      # Environment variables
```

**Why these packages?**
- **Flask:** Lightweight web framework for REST API
- **CORS:** Allow requests from different domains (Flutter app)
- **scikit-learn:** KNN and preprocessing algorithms
- **pandas:** Read and manipulate CSV dataset
- **numpy:** Numerical operations
- **joblib:** Save and load ML models efficiently
- **python-dotenv:** Manage environment configuration

---

## 🚀 Production Deployment

### Using Gunicorn (Production WSGI Server)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn app:app --workers 4 --bind 0.0.0.0:5000

# Expected output:
# [INFO] Server is ready. Spawning workers
# [INFO] Spawning worker with pid: xxxxx
```

### Using Docker

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

**Build and Run:**
```bash
docker build -t smart-hospital-ml .
docker run -p 5000:5000 smart-hospital-ml
```

### Deploy to Cloud Platform

**Heroku:**
```bash
heroku create smart-hospital-ml
git push heroku main
```

**Google Cloud Run:**
```bash
gcloud run deploy smart-hospital-ml --source . --platform managed
```

**AWS EC2:**
```bash
# SSH into instance
# Clone repository
# Set up environment
# Run with Gunicorn
```

---

## 🔐 Security Considerations

1. **Input Validation**
   - All input fields are validated
   - Type checking and range validation
   - Missing field detection

2. **Error Handling**
   - Graceful error responses
   - No sensitive information in error messages
   - Proper HTTP status codes

3. **CORS Configuration**
   - Configured to allow Flutter app requests
   - Restrict in production to specific domains

4. **Model Security**
   - Models stored safely on disk
   - Access controlled by file permissions
   - No model exposure via API

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Change port in .env
FLASK_PORT=5001

# Or kill the process using port 5000
# Windows: netstat -ano | findstr :5000
# macOS/Linux: lsof -i :5000
```

### Issue: "Models directory not found"

**Solution:**
```bash
python train_model.py
```

This creates the models directory and trains the model.

### Issue: "Prediction returns unexpected result"

**Possible causes:**
1. Input values outside expected ranges
2. Model not trained properly
3. Scaler not applied correctly

**Solution:**
```bash
# Retrain the model
python train_model.py

# Check input values are reasonable
```

---

## 📊 Extending the Model

### Adding New Training Data

1. Edit `data/medical_dataset.csv`
2. Add new rows with clinical data and diagnosis
3. Run `python train_model.py` to retrain

### Using Different Algorithm

Replace KNN with another algorithm:

```python
from sklearn.ensemble import RandomForestClassifier

# Replace KNN
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

### Adding More Features

1. Collect more clinical variables
2. Add columns to CSV
3. Update form in Flutter app
4. Retrain the model

---

## 📝 API Usage Examples

### Example 1: JavaScript/Fetch API

```javascript
async function getDiagnosis() {
  const data = {
    age: 35,
    gender: 1,
    temperature: 37.2,
    blood_pressure_systolic: 120,
    blood_pressure_diastolic: 80,
    heart_rate: 72,
    glucose: 95,
    hemoglobin: 13.5,
    white_blood_cells: 7.2,
    platelets: 250
  };

  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  console.log(result);
}
```

### Example 2: Python/Requests

```python
import requests

BASE_URL = 'http://localhost:5000'
data = {
    'age': 35,
    'gender': 1,
    'temperature': 37.2,
    'blood_pressure_systolic': 120,
    'blood_pressure_diastolic': 80,
    'heart_rate': 72,
    'glucose': 95,
    'hemoglobin': 13.5,
    'white_blood_cells': 7.2,
    'platelets': 250
}

response = requests.post(f'{BASE_URL}/predict', json=data)
result = response.json()
print(result)
```

### Example 3: Dart/Flutter

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<void> getPrediction() async {
  final data = {
    'age': 35,
    'gender': 1,
    'temperature': 37.2,
    'blood_pressure_systolic': 120,
    'blood_pressure_diastolic': 80,
    'heart_rate': 72,
    'glucose': 95,
    'hemoglobin': 13.5,
    'white_blood_cells': 7.2,
    'platelets': 250,
  };

  final response = await http.post(
    Uri.parse('http://localhost:5000/predict'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(data),
  );

  if (response.statusCode == 200) {
    final result = jsonDecode(response.body);
    print(result);
  }
}
```

---

## 📚 References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [REST API Best Practices](https://restfulapi.net/)
- [K-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

## 📄 License

Part of Smart Hospital academic project.

---

## ✨ Notes

- This is a demonstration/educational system
- Not suitable for real medical decision-making
- Always consult qualified healthcare professionals
- Model performance depends on training data quality

---

**Last Updated:** December 2025  
**Version:** 1.0.0
