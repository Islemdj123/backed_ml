import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Optional: load environment variables for flexible configuration
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -----------------------------------------------------------------------------
DATA_SOURCE = os.getenv('DATA_SOURCE', 'csv').lower()  # 'csv' or 'mongodb'
DATA_PATH = os.getenv('DATA_PATH', 'data/final_medical_dataset_10000.csv')

# MongoDB configuration (used only if DATA_SOURCE == 'mongodb')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'smart_hospital')
MONGO_COLLECTION = os.getenv('MONGO_COLLECTION', 'medical_dataset')
MONGO_QUERY = os.getenv('MONGO_QUERY')  # Optional JSON filter

# Model artifact paths
MODEL_DIR = 'models'
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(MODEL_DIR, 'knn_model.pkl'))
SCALER_PATH = os.getenv('SCALER_PATH', os.path.join(MODEL_DIR, 'scaler.pkl'))
LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the training dataset from CSV or MongoDB.

    Expected columns (features + target):
    [
        'age', 'gender', 'temperature', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'heart_rate', 'glucose', 'hemoglobin',
        'white_blood_cells', 'platelets', 'diagnosis'
    ]
    """
    if DATA_SOURCE == 'mongodb':
        try:
            from pymongo import MongoClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                'pymongo is required for MongoDB data source. Install it and retry.'
            ) from e

        print('Loading medical dataset from MongoDB...')
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        col = db[MONGO_COLLECTION]

        query = {}
        if MONGO_QUERY:
            import json
            try:
                query = json.loads(MONGO_QUERY)
            except Exception as e:
                raise ValueError('Invalid MONGO_QUERY. Must be valid JSON.') from e

        docs = list(col.find(query, {'_id': 0}))
        if not docs:
            raise ValueError('No documents found in MongoDB collection for the given query.')

        df = pd.DataFrame(docs)
        print(f"MongoDB dataset shape: {df.shape}")
        return df

    # Default: CSV
    print('Loading medical dataset from CSV...')
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Dataset not found at {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    
    # Rename columns to match internal schema
    column_mapping = {
        'Glucose': 'glucose',
        'Cholesterol': 'cholesterol',
        'Hemoglobin': 'hemoglobin',
        'Platelets': 'platelets',
        'White Blood Cells': 'white_blood_cells',
        'Red Blood Cells': 'red_blood_cells',
        'Hematocrit': 'hematocrit',
        'Mean Corpuscular Volume': 'mean_corpuscular_volume',
        'Mean Corpuscular Hemoglobin': 'mean_corpuscular_hemoglobin',
        'Mean Corpuscular Hemoglobin Concentration': 'mean_corpuscular_hemoglobin_concentration',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'Systolic Blood Pressure': 'blood_pressure_systolic',
        'Diastolic Blood Pressure': 'blood_pressure_diastolic',
        'Triglycerides': 'triglycerides',
        'HbA1c': 'hba1c',
        'LDL Cholesterol': 'ldl_cholesterol',
        'HDL Cholesterol': 'hdl_cholesterol',
        'ALT': 'alt',
        'AST': 'ast',
        'Heart Rate': 'heart_rate',
        'Creatinine': 'creatinine',
        'Troponin': 'troponin',
        'C-reactive Protein': 'c_reactive_protein',
        'Disease': 'diagnosis'
    }
    
    # Check if we are using the final_medical_dataset_10000 format
    if 'RBC' in df.columns and 'WBC' in df.columns:
        print("Detected final_medical_dataset_10000 format. Renaming columns...")
        new_mapping = {
             'Hemoglobin': 'hemoglobin',
             'RBC': 'red_blood_cells',
             'WBC': 'white_blood_cells',
             'Platelets': 'platelets',
             'MCV': 'mean_corpuscular_volume',
             'Glucose': 'glucose',
             'Cholesterol': 'cholesterol',
             'Triglycerides': 'triglycerides',
             'Creatinine': 'creatinine',
             'Urea': 'urea',
             'CRP': 'c_reactive_protein',
             'Blood_Pressure': 'blood_pressure',
             'Heart_Rate': 'heart_rate',
             'BMI': 'bmi',
             'Sodium': 'sodium',
             'Potassium': 'potassium',
             'Calcium': 'calcium',
             'Diagnosis': 'diagnosis'
        }
        df = df.rename(columns=new_mapping)
        # Ensure all expected columns are present
        expected_columns = list(new_mapping.values())
        df = df[expected_columns]

    # Check if we are using the new dataset format (old one)
    elif 'Glucose' in df.columns and 'Disease' in df.columns:
        print("Detected old dataset format. Renaming columns...")
        df = df.rename(columns=column_mapping)
        
        # Ensure all expected columns are present
        expected_columns = list(column_mapping.values())
        # Filter to keep only the expected columns (in case there are extras)
        df = df[expected_columns]
        
    print(f"CSV dataset shape: {df.shape}")
    return df


# -----------------------------------------------------------------------------
# Training Pipeline (K-Nearest Neighbors)
# -----------------------------------------------------------------------------

def main():
    print('Starting training pipeline (KNN)...')

    df = load_data()

    # Verify we have all features
    # Determine required columns based on dataset columns
    if 'blood_pressure' in df.columns:
        required_columns = [
            'hemoglobin', 'red_blood_cells', 'white_blood_cells', 'platelets',
            'mean_corpuscular_volume', 'glucose', 'cholesterol', 'triglycerides',
            'creatinine', 'urea', 'c_reactive_protein', 'blood_pressure',
            'heart_rate', 'bmi', 'sodium', 'potassium', 'calcium', 'diagnosis'
        ]
    else:
        required_columns = [
            'glucose', 'cholesterol', 'hemoglobin', 'platelets', 'white_blood_cells',
            'red_blood_cells', 'hematocrit', 'mean_corpuscular_volume',
            'mean_corpuscular_hemoglobin', 'mean_corpuscular_hemoglobin_concentration',
            'insulin', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'triglycerides', 'hba1c', 'ldl_cholesterol', 'hdl_cholesterol',
            'alt', 'ast', 'heart_rate', 'creatinine', 'troponin', 'c_reactive_protein',
            'diagnosis'
        ]
    
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    print(f"Features: {df.columns.tolist()}")
    print(f"Diagnosis distribution:\n{df['diagnosis'].value_counts()}")

    # Filter out classes with too few samples (need at least 2 for stratified split)
    class_counts = df['diagnosis'].value_counts()
    min_samples = 2
    valid_classes = class_counts[class_counts >= min_samples].index
    dropped_classes = class_counts[class_counts < min_samples].index
    
    if not dropped_classes.empty:
        print(f"\nDropping classes with fewer than {min_samples} samples: {dropped_classes.tolist()}")
        df = df[df['diagnosis'].isin(valid_classes)]
        print(f"Filtered dataset shape: {df.shape}")

    # Split features/target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nClass mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")

    # KNN classifier with Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print('Model Performance Metrics:')
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*50}")

    confusion = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{confusion}")

    # Persist artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Label Encoder saved to: {LABEL_ENCODER_PATH}")
    print('\nTraining completed successfully!')


if __name__ == '__main__':
    main()
