from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Debug: Print versions
import sklearn
print(f"Python version: {os.sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Load the trained model
model = None
try:
    print("Loading model...")
    model = joblib.load('heart_disease_model.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Heart Disease Prediction API is running!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded. Check server logs.'})
        
        # Get form data
        data = request.form
        
        print(f"Received data: {dict(data)}")
        
        # Create DataFrame with correct dtypes
        input_data = pd.DataFrame([[
            float(data['age']),
            data['sex'],
            data['chestPainType'],
            float(data['restingBP']),
            float(data['cholesterol']),
            int(data['fastingBS']),
            data['restingECG'],
            float(data['maxHR']),
            data['exerciseAngina'],
            float(data['oldpeak']),
            data['stSlope']
        ]], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                    'Oldpeak', 'ST_Slope'])
        
        # Apply Data Cleaning and Feature Engineering to match improved model
        input_data['Cholesterol'] = input_data['Cholesterol'].replace(0, np.nan)
        input_data['RestingBP'] = input_data['RestingBP'].replace(0, np.nan)
        input_data['AgeCategory'] = pd.cut(input_data['Age'], bins=[0, 40, 60, 150], labels=['<40', '40-60', '>60'])
        input_data['CholesterolRatio'] = input_data['Cholesterol'] / 200.0
        input_data['RestingBPRatio'] = input_data['RestingBP'] / 120.0
        
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        
        print(f"Prediction: {prediction}, Probability: {probability}")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk': 'High' if prediction == 1 else 'Low',
            'message': 'Heart disease risk detected. Please consult a cardiologist.' if prediction == 1 else 'No significant risk detected. Maintain healthy lifestyle.'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'python_version': os.sys.version.split()[0]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
