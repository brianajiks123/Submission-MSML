import json, time, sys, requests

# --------------------------
# CONFIGURATION
# --------------------------

EXPORTER_URL = "http://localhost:8001"
PREDICT_ENDPOINT = f"{EXPORTER_URL}/predict"
METRICS_ENDPOINT = f"{EXPORTER_URL}/metrics"
INPUT_FILE = "input.json"

# --------------------------
# FUNCTIONS
# --------------------------

def health_check_predict():
    print("== Health Check Exporter Predict ==")
    
    try:
        with open(INPUT_FILE, "r") as f:
            payload = json.load(f)

        response = requests.post(PREDICT_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            print("[OK] Exporter /predict OK (200)")
        else:
            print(f"[FAIL] /predict returned HTTP {response.status_code}")
            print(f"       Response body: {response.text}")
            
            exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to check /predict: {e}")
        
        exit(1)

def run_load_test(n=1000):
    print(f"\n== Starting load test: {n} requests → {PREDICT_ENDPOINT} ==")
    
    with open(INPUT_FILE, "r") as f:
        payload = json.load(f)

    errors = 0
    start_time = time.time()

    for i in range(1, n + 1):
        try:
            res = requests.post(PREDICT_ENDPOINT, json=payload)
            
            if res.status_code != 200:
                errors += 1
        except:
            errors += 1

        if i % 100 == 0:
            elapsed = time.time() - start_time
            
            print(f" → {i} requests done in {elapsed:.1f}s (errors={errors})")

    total_time = time.time() - start_time
    
    print(f"\nLoad test completed in {total_time:.1f}s, total errors: {errors}")

def fetch_metrics():
    print("\n== Fetching Prometheus metrics ==")
    
    try:
        res = requests.get(METRICS_ENDPOINT)
        
        if res.status_code == 200:
            text = res.text
            
            for line in text.splitlines():
                if line.startswith("#") or line.strip() == "":
                    continue
                
                print(line)
        else:
            print(f"[FAIL] Failed to fetch /metrics: HTTP {res.status_code}")
    except Exception as e:
        print(f"[ERROR] Exception while fetching /metrics: {e}")


if __name__ == "__main__":
    health_check_predict()
    run_load_test(n=10000)
    fetch_metrics()
