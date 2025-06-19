# Monitoring & Alerting Obesity Classification Model

## Query Metrics Prometheus

- rate(http_requests_total[5m])
- rate(http_requests_throughput_total[5m])
- rate(http_request_errors_total[5m])
- histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
- avg(rate(http_request_size_bytes_sum[5m])) / avg(rate(http_request_size_bytes_count[5m]))
- sum(rate(http_request_size_bytes_bucket[5m])) by (le)
- system_cpu_usage_percent
- system_ram_usage_percent
- sum(rate(input_age_distribution_bucket[5m])) by (le)
- sum(rate(predictions_by_class_total[5m])) by (obesity_class)
