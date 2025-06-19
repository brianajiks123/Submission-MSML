import pandas as pd, joblib, requests, json

preprocessor = joblib.load("model/obesity_preprocessor.joblib")
label_encoder = joblib.load("model/obesity_label_encoder.joblib")

PREDICTION_URL = "http://localhost:5005/invocations"

FEATURE_NAMES = [
    name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()
]

RAW_COLUMNS = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC',
    'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
]


def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict], columns=RAW_COLUMNS)
    X_proc = preprocessor.transform(df)
    
    return X_proc


def predict_obesity(data_dict):
    X_proc = preprocess_input(data_dict)
    
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": X_proc.tolist()
        }
    }
    
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(PREDICTION_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_msg = response.text
        
        raise RuntimeError(f"Prediction API error: HTTP {response.status_code} - {error_msg}")

    preds = response.json().get("predictions")
    pred_int = int(preds[0])
    label = label_encoder.inverse_transform([pred_int])[0]
    
    return label


if __name__ == "__main__":
    try:
        with open('input.json', 'r') as f:
            sample_input = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("input.json not found. Please create the file with sample feature values.")

    result = predict_obesity(sample_input)
    
    print("Predicted Obesity Category:", result)
