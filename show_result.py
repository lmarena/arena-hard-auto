import pandas as pd
import argparse
import os
import re
import math
import numpy as np
from glob import glob

pattern = re.compile(r"\[\[(\d*?)\]\]")

def evaluate(df):
    scores = []
    for i, row in df.iterrows():
        if not math.isnan(row["score"]):
            scores.append(int(row["score"]))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-bench-v1")
    args = parser.parse_args()
    print(args)

    model_scores = {}
    for judgment_file in glob(os.path.join("data", args.bench_name, "model_judgment", "*.jsonl")):
        df = pd.read_json(judgment_file, lines=True)
        assert len(df["model"].unique()) == 1
        model_scores[df.iloc[0]["model"]] = evaluate(df)

    keys = list(model_scores.keys())
    values = list(model_scores.values())
    value_sums = [-1 * sum(v) / len(v) for v in values]
    sorted_value_index = np.argsort(value_sums)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

    for model_name, scores in sorted_dict.items():
        print(f"Model: {model_name : <20} | Score: {sum(scores) / len(scores) : ^20} | # Valid-score: {len(scores) : >2}")