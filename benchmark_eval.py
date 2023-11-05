import pandas as pd
import argparse
import os
import re
from glob import glob

pattern = re.compile(r"\[\[(\d*?)\]\]")

def evaluate(df):
    model_names = df["model_id"].unique()
    model_scores = {}
    for model_name in model_names:
        scores = []
        for i, row in df.iterrows():
            if row["model_id"] != model_name:
                continue
            matches = pattern.findall(row["judgment"][0]["turns"][0])
            # remove empty string
            matches = [m for m in matches if m != ""]
            if len(matches) == 0:
                print("WARNING: no score pattern matched... skipping "
                      f"(model={row['model_id']}, qid={row['question_id']})")
            elif len(matches) == 1:
                scores.append(int(matches[0]))
            else:
                print("WARNING: more than one score pattern matched")
        model_scores[model_name] = scores

    for model_name, scores in model_scores.items():
        print(f"Model: {model_name} | Score: {sum(scores) / len(scores)} | # Valid-score: {len(scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, required=True)
    args = parser.parse_args()
    print(args)

    for judgment_file in glob(os.path.join("data", args.bench_name, "model_judgment", "*.jsonl")):
        evaluate(pd.read_json(judgment_file, lines=True))
