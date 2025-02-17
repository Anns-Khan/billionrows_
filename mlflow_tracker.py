import mlflow
import yaml
import os

class MLflowTracker:
    """
    A reusable MLflow tracking class that reads configuration from a YAML file.
    This class sets up the MLflow experiment and run, and exposes methods to log parameters,
    metrics, and artifacts.
    """
    def __init__(self, config_file="mlflow_config.yaml"):
        # Load configuration from the YAML file.
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set up configuration parameters
        self.experiment_name = self.config.get("experiment_name", "Default_Experiment")
        self.run_name = self.config.get("run_name", "Default_Run")
        self.tracking_uri = self.config.get("tracking_uri", None)
        self.metrics_to_track = self.config.get("metrics", [])
        
        # Set tracking URI if provided
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set the experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Start the run
        self.run = mlflow.start_run(run_name=self.run_name)
        print(f"MLflow run started: Experiment='{self.experiment_name}', Run='{self.run_name}'")
    
    def log_param(self, key, value):
        """Log a single parameter."""
        mlflow.log_param(key, value)
    
    def log_params(self, params: dict):
        """Log multiple parameters from a dictionary."""
        mlflow.log_params(params)
    
    def log_metric(self, key, value, step=None):
        """Log a single metric. Optionally include a step value."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: dict, step=None):
        """Log multiple metrics from a dictionary."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)
    
    def log_artifact(self, file_path, artifact_path=None):
        """Log an artifact (file) to MLflow."""
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print("MLflow run ended.")

# Example usage (can be removed or placed in a separate test script):
if __name__ == "__main__":
    tracker = MLflowTracker("mlflow_config.yaml")
    tracker.log_param("example_param", 123)
    tracker.log_metric("example_metric", 0.95)
    tracker.end_run()
