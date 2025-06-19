import requests, time, psutil, os
from dotenv import load_dotenv
from joblib import load
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

app = Flask(__name__)

# ----- API Metrics -----
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests received'
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'Latency of HTTP requests in seconds'
)
THROUGHPUT = Counter(
    'http_requests_throughput',
    'Number of requests processed'
)
REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'Size of HTTP request payloads in bytes'
)
ERROR_COUNT = Counter(
    'http_request_errors_total',
    'Total number of errors during request handling'
)

# ----- System Metrics -----
CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)
RAM_USAGE = Gauge(
    'system_ram_usage_percent',
    'System RAM usage percentage'
)

# ----- Input Data Metrics -----
AGE_HISTOGRAM = Histogram(
    'input_age_distribution',
    'Histogram of input ages',
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)
BMI_HISTOGRAM = Histogram(
    'input_bmi_distribution',
    'Histogram of input BMI values',
    buckets=[15, 18.5, 25, 30, 35, 40, 45, 50]
)

# ----- Prediction Metrics -----
PREDICTION_COUNTER = Counter(
    'predictions_by_class',
    'Count of predictions labeled by obesity class',
    ['obesity_class']
)

# Model API URL (inside Docker on Windows)
MODEL_API_URL = os.environ.get(
    'MODEL_API_URL',
    'http://host.docker.internal:5005/invocations'
)


# Load Preprocessor & Label Encoder
PREPROCESSOR = load("model/obesity_preprocessor.joblib")
FEATURE_NAMES = [ name.split("__", 1)[1] for name in PREPROCESSOR.get_feature_names_out() ]
LABEL_ENCODER = load("model/obesity_label_encoder.joblib")


@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    try:
        payload_size = len(request.data or b"")
        
        REQUEST_SIZE.observe(payload_size)

        data = request.get_json()
        
        # Observe Input Metrics
        age = float(data.get('Age', 0))
        
        AGE_HISTOGRAM.observe(age)
        
        height = float(data.get('Height', 1))
        weight = float(data.get('Weight', 0))
        bmi = weight / ((height/100) ** 2)
        
        BMI_HISTOGRAM.observe(bmi)

        # Wrap Payload for Model
        payload = {
            "dataframe_split": {
                "columns": FEATURE_NAMES,
                "data": [
                    [ data.get(col, 0) for col in FEATURE_NAMES ]
                ]
            }
        }

        # Call Model
        response = requests.post(MODEL_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # Decode Int Prediction to Label
        pred_int = int(result["predictions"][0])
        obesity_class = LABEL_ENCODER.inverse_transform([pred_int])[0]

        # Increment Prediction Counter with Label
        PREDICTION_COUNTER.labels(obesity_class=obesity_class).inc()
    except Exception as e:
        ERROR_COUNT.inc()
        
        return jsonify({"error": str(e)}), 500
    finally:
        duration = time.time() - start_time
        
        REQUEST_LATENCY.observe(duration)

    return jsonify(result)


if __name__ == '__main__':
    host = os.environ.get('EXPORTER_HOST', '0.0.0.0')
    port = int(os.environ.get('EXPORTER_PORT', 8001))
    
    app.run(host=host, port=port)
