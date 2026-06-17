"""Orchestrator — runs all four pipeline stages in order under a
single MLflow run.

Logs the config-driven model hyperparameters as run parameters
before the stages fire, and the final evaluation metrics (read back
from `reports/evaluation.json`) once the last stage completes. Fails
fast on the first non-zero stage exit.
"""
import json
import os
import subprocess
import sys

import mlflow
import yaml

os.chdir("/root/code/fraud-detection")

with open("configs/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

STAGES = ["preprocess.py", "featurize.py", "train.py", "evaluate.py"]


def main():
    with mlflow.start_run(run_name="full-pipeline"):
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])

        for stage in STAGES:
            print(f"[pipeline] running src/{stage} ...")
            result = subprocess.run(
                [sys.executable, f"src/{stage}"],
                capture_output=True, text=True,
            )
            sys.stdout.write(result.stdout)
            if result.returncode != 0:
                sys.stderr.write(result.stderr)
                raise SystemExit(f"[pipeline] stage failed: {stage}")

        report_path = config["output"]["report_path"]
        with open(report_path) as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(config["output"]["model_path"])
        print("[pipeline] completed.")


if __name__ == "__main__":
    main()