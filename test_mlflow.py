import mlflow

# Set experiment name
mlflow.set_experiment("test_experiment")

# Start a new run
with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.5)

