import argparse, os, mlflow, mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def setup_mlflow_local(local_uri=None):
    if local_uri:
        mlflow.set_tracking_uri(local_uri)
    
    print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

def train_and_log_autolog(X_train, X_test, y_train, y_test, random_state):
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_RF_Autolog"):
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        
        print(f"🔢 Test accuracy: {acc:.4f}")

        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(model, "models/baseline_rf_autolog")
        
        print("✅ Model saved to models/baseline_rf_autolog")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Modelling with MLflow before tuning")
    
    parser.add_argument("--input", required=True, help="Path to Obesity_preprocessed.csv")
    parser.add_argument("--mode", choices=["local", "online"], default="local", help="Tracing mode MLflow")
    parser.add_argument("--local_uri", help="local URI MLflow")
    parser.add_argument("--repo_owner", help="Owner DagsHub repository (for online mode)")
    parser.add_argument("--repo_name", help="DagsHub repository name (for online mode)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set Proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for split")
    
    args = parser.parse_args()

    # Setup MLflow
    setup_mlflow_local(args.local_uri)

    # Load data
    df = pd.read_csv(args.input)
    X = df.drop("Obesity", axis=1)
    y = df["Obesity"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # Train & log with autolog
    model = train_and_log_autolog(
        X_train, X_test, y_train, y_test,
        random_state=args.random_state
    )

    print("✅ Finish baseline run with MLflow Autolog.")
