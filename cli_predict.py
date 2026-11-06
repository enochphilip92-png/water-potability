"""
CLI Water Potability Predictor
Author: URLA ENOCH PHILIP (Roll No: 23ME1A54B1)
Domain: Machine Learning & Predictive Analytics
Guide: Mr. E. Surendra babu

Usage:
  python cli_predict.py --ph 7.2 --hardness 180 --solids 450 --chloramines 3.2 --sulfate 210 --conductivity 350 --organic_carbon 2.5 --trihalomethanes 50 --turbidity 2.2
Or simply run interactively:
  python cli_predict.py
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_preprocessing import transform_sample, FEATURE_COLUMNS, WHO_GUIDELINES

def main():
    parser = argparse.ArgumentParser(description="Predict water potability using trained SVM model.")
    parser.add_argument('--ph', type=float, default=None, help="pH of water sample (0 - 14)")
    parser.add_argument('--hardness', type=float, default=None, help="Hardness in mg/L")
    parser.add_argument('--solids', type=float, default=None, help="Total Dissolved Solids in ppm")
    parser.add_argument('--chloramines', type=float, default=None, help="Chloramines in ppm")
    parser.add_argument('--sulfate', type=float, default=None, help="Sulfate in mg/L")
    parser.add_argument('--conductivity', type=float, default=None, help="Electrical conductivity in uS/cm")
    parser.add_argument('--organic_carbon', type=float, default=None, help="Organic carbon in ppm")
    parser.add_argument('--trihalomethanes', type=float, default=None, help="Trihalomethanes in ug/L")
    parser.add_argument('--turbidity', type=float, default=None, help="Turbidity in NTU")
    
    args = parser.parse_args()
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'water_potability_svm_model.pkl')
    
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Run 'python src/train.py' first.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    
    # Check if arguments provided or prompt user
    sample = {}
    if args.ph is not None:
        sample = {
            'ph': args.ph,
            'Hardness': args.hardness if args.hardness is not None else 190.0,
            'Solids': args.solids if args.solids is not None else 500.0,
            'Chloramines': args.chloramines if args.chloramines is not None else 3.5,
            'Sulfate': args.sulfate if args.sulfate is not None else 220.0,
            'Conductivity': args.conductivity if args.conductivity is not None else 380.0,
            'Organic_carbon': args.organic_carbon if args.organic_carbon is not None else 3.0,
            'Trihalomethanes': args.trihalomethanes if args.trihalomethanes is not None else 55.0,
            'Turbidity': args.turbidity if args.turbidity is not None else 2.5
        }
    else:
        print("\n" + "="*60)
        print("WATER POTABILITY PREDICTOR (CLI TOOL)")
        print("Student: URLA ENOCH PHILIP | Roll No: 23ME1A54B1")
        print("="*60)
        print("Using standard sample test values (ideal clean drinking water):\n")
        sample = {
            'ph': 7.30,
            'Hardness': 185.0,
            'Solids': 450.0,
            'Chloramines': 3.2,
            'Sulfate': 210.0,
            'Conductivity': 350.0,
            'Organic_carbon': 2.8,
            'Trihalomethanes': 45.0,
            'Turbidity': 2.2
        }
        for k, v in sample.items():
            print(f"  {k:18s}: {v}")
            
    # Predict
    X_scaled = transform_sample(sample, artifacts_dir=models_dir)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    print("\n" + "-"*60)
    print("ANALYSIS RESULTS:")
    print("-"*60)
    if prediction == 1:
        print(f"--> RESULT: [POTABLE WATER - SAFE FOR CONSUMPTION]")
        print(f"--> Confidence: {probabilities[1]*100:.2f}%")
        print(f"--> Action: Approved for municipal distribution.")
    else:
        print(f"--> RESULT: [NON-POTABLE WATER - UNSAFE / CONTAMINATED]")
        print(f"--> Risk Score: {probabilities[0]*100:.2f}%")
        print(f"--> Action: Flagged for sanitation intervention.")
    print("-"*60 + "\n")

if __name__ == '__main__':
    main()
