# Submission MSML 🚀

## Requirements 📋

- Python v3.12.7 🐍
- MLFlow 📊
- DagsHub 🌐
- Prometheus 📈
- Grafana 📉

## Screenshots 🖼️



## Setup & Run 🛠️

1. **Create & Activate Virtual Environment** (Recommended) 🌍

    - Windows 10/11

        ``` python
        python -m venv venv
        ```

    - Linux or Mac 🖥️

        ``` bash
        source venv/bin/activate
        ```

2. **Install Library** 📦

    ``` python
    pip install -r requirements.txt
    ```

3. **Modeling** 🤖

    - Local Mode (Default) 🏠

        ``` python
        python train_model_file.py --input path/to/Obesity_preprocessed.csv --mode local
        ```

    - Using Local URI 🔗

        ``` python
        python train_model_file.py --input path/to/Obesity_preprocessed.csv --mode local --local_uri http://localhost:5000
        ```

    - Using All Arguments (Optional) ⚙️

        ``` python
        python train_model_file.py --input data/Obesity_preprocessed.csv --mode local --test_size 0.25 --random_state 100
        ```

    - Online Mode (DagsHub) ☁️

        ``` python
        python train_model_file.py --input path/to/Obesity_preprocessed.csv --mode online --repo_owner your_username --repo_name your_repo_name
        ```

4. View MLflow Logs 📜

    ``` bash
    mlflow ui
    ```

5. Create Monitoring & Alerting 🔍

    - Run Prometheus 📈
    - Run Prometheus Exporter 🚀

        ``` python
        python prometheus_exporter.py
        ```

    - Try Inference 🧠

        ``` python
        python inference.py
        ```

        OR

        ``` python
        python test_inference_and_metrics.py
        ```

    - View Metrics 📊

        ``` bash
        http://localhost:8001/metrics
        ```

    - Login Prometheus for trying query 🔎
    - Login Grafana for create monitoring dashboard 📉
