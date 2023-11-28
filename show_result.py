import pandas as pd
import argparse
import os
import re
import math
import numpy as np
import plotly.express as px
from glob import glob

pattern = re.compile(r"\[\[(\d*?)\]\]")

def evaluate(df):
    scores = []
    for i, row in df.iterrows():
        if not math.isnan(row["score"]):
            scores.append(int(row["score"]))
    return scores, df.iloc[0]["model"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-bench-v1")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    print(args)

    model_scores = {}
    models = []
    scores = [] 
    for judgment_file in glob(os.path.join("data", args.bench_name, "model_judgment", "*.jsonl")):
        df = pd.read_json(judgment_file, lines=True)
        assert len(df["model"].unique()) == 1
        score, model = evaluate(df)
        models.append(model)
        scores.append(score)

    leaderboard = pd.DataFrame()
    leaderboard["models"] = models
    leaderboard["scores"] = [sum(s) / len(s) for s in scores]
    leaderboard["valid"] = [len(s) for s in scores]
    leaderboard = leaderboard.sort_values(by="scores", ascending=False)
    # print(leaderboard.to_string(index=False))
    for i, row in leaderboard.iterrows():
        print(f"Model: {row['models'] : <25} | Score: {row['scores'] : ^20} | # Valid-score: {row['valid'] : >2}")

    if args.plot:
        fig = px.bar(leaderboard, x="scores", y="models", orientation='h', color='models')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})

        scores = np.round(leaderboard["scores"].to_list(), decimals=2)
        annotations = []
        for score, x in zip(scores, leaderboard["models"].to_list()):
            annotations.append(dict(xref='x1', yref='y1',
                            y=x, x=score + 0.2,
                            text=str(score),
                            font=dict(family='Arial', size=20),
                            showarrow=False))
        fig.update_layout(annotations=annotations)
        fig.show()