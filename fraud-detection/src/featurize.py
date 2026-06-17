"""Stage 2 — Featurize.

Reads the upstream stage's output, engineers one derived column
(`amount_log = log1p(amount)`), and writes the feature matrix to
`data/features/features.csv` for the training stage to consume.

Every concern other than the input wiring is correctly in place —
feature engineering, column preservation, on-disk layout. Adjust
the input source so the stage-chain invariant holds: the row count
out of this stage must match the row count the preprocess stage
produced.
"""
import os

import numpy as np
import pandas as pd
import yaml

os.chdir("/root/code/fraud-detection")

with open("configs/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

input_path = config["data"]["processed_path"]
features_path = config["data"]["features_path"]

df = pd.read_csv(input_path)
df["amount_log"] = np.log1p(df["amount"])

os.makedirs(os.path.dirname(features_path), exist_ok=True)
df.to_csv(features_path, index=False)

print(
    f"[featurize] input={input_path}  rows={len(df)}  "
    f"columns={len(df.columns)}"
)