import joblib
import os
import pandas as pd
import json
import numpy as np
import sys
import imblearn
from io import BytesIO, StringIO


model_dir = os.environ.get('SM_MODEL_DIR') 

if model_dir not in sys.path:
    sys.path.append(model_dir)

from src.Custom_Classes import FeatureEngineer

def model_fn(model_dir):
    """Load the model from the specified directory."""
    path = os.path.join(model_dir, 'finalized_loan_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    model = joblib.load(path)
    print("Model loaded successfully.")
    return model

def input_fn(request_body, request_content_type):
    print(f"Receiving data of type: {request_content_type}")
    
    # 1. Handle the default SageMaker serialization
    if request_content_type == 'application/x-npy':
        # request_body is a stream of bytes in .npy format
        data = np.load(BytesIO(request_body), allow_pickle=True)
        # If your model needs a DataFrame, convert it here
        return pd.DataFrame(data)
    
    elif request_content_type == 'application/json':
        return pd.read_json(request_body)
    
    elif request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_df, model):
    """Apply the model to the parsed data."""
    print("Running prediction pipeline...")
    return model.predict(input_df)

def output_fn(prediction, content_type):
    """
    Standard function to format the output.
    Converts NumPy predictions back to JSON for the client.
    """
    print("Formatting output...")
    res = prediction.tolist() if isinstance(prediction, (np.ndarray, np.generic)) else prediction
    return json.dumps(res), "application/json"

