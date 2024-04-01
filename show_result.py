import pandas as pd
import argparse
import os
import numpy as np

import tiktoken
import datetime

from glob import glob
from utils import load_model_answers

label_map = {"A":"A>>B", "B":"A>B", "C":"A=B", "D":"B>A", "E":"B>>A"}
win_map = {"A":"L", "B":"L", "C":"T", "D":"W", "E":"W"}
win_weight = {"A":3, "B":1, "C":1, "D":1, "E":3}

reverse_map = {"A":"E", "B":"D", "C":"C", "D":"B", "E":"A"}

new_grading = {"W": 100, "T":50, "L":0}
old_grading = {"A":-3, "B":-1, "C":0, "D":1, "E":3}

def evaluate(df):
    scores = {"W":0, "L":0, "T":0}
    stats = {}
    stats["valid"] = 0

    for label in old_grading.keys():
        stats[label] = 0

    for _, row in df.iterrows():

        game = row["games"][0]
        if game["score"] is not None:
            stats["valid"] += 1
            for label in win_map.keys():
                if game["score"] == label_map[label]:
                    scores[win_map[label]] += win_weight[label]
                    stats[label] += 1
                    break
                
        # Game 2
        game = row["games"][1]
        if game["score"] is not None:
            stats["valid"] += 1
            for label in win_map.keys():
                if game["score"] == label_map[label]:
                    scores[win_map[reverse_map[label]]] += win_weight[reverse_map[label]]
                    stats[reverse_map[label]] += 1
                    break

    stats["rate"] = 0
    for label in new_grading.keys():
        stats["rate"] += scores[label] * new_grading[label]
    stats["rate"] = np.round(stats["rate"] / sum(scores.values()), decimals=2)

    return stats, df.iloc[0]["model"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--full-stats", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--output", action="store_true")
    args = parser.parse_args()
    print(args)

    win_weight = {"A":args.weight, "B":1, "C":1, "D":1, "E":args.weight}

    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    model_answers = load_model_answers(answer_dir)
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

    models = []
    scores = []
    lengths = []
    for judgment_file in glob(os.path.join("data", args.bench_name, f"model_judgment/{args.judge_name}", "*.jsonl")):
        df = pd.read_json(judgment_file, lines=True)
        assert len(df["model"].unique()) == 1
        score, model = evaluate(df)
        models.append(model)
        scores.append(score)

        length = 0
        if model in model_answers:
            for id, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                length += turn["token_len"]
            length /= len(model_answers[model])
        else:
            print(f"Cannot find {model} in model_answer directory.")
        lengths.append(int(length))
        
    leaderboard = pd.DataFrame()
    leaderboard["model"] = models
    leaderboard["winrate"] = [s["rate"] for s in scores]
    leaderboard["stats"] = scores
    leaderboard["avg_token"] = lengths
    leaderboard = leaderboard.sort_values(by="winrate", ascending=False)
    for i, row in leaderboard.iterrows():
        if args.full_stats:
            print(f"{row['model'][:20] : <20} | win-rate: {row['winrate'] : ^5} | vaild-score: {row['stats']['valid'] : ^3} | big-win: {row['stats']['E'] : ^2} | small-win: {row['stats']['D']} | tie: {row['stats']['C'] : ^2} | big-loss: {row['stats']['A'] : ^2} | small-loss: {row['stats']['B']}")
        else:
            print(f"{row['model'] : <30} | win-rate: {row['winrate'] : ^5} | average #tokens: {row['avg_token']}")
        # print(f"[{row['model']}, {row['scores']}: {row['avg_token']}]")

    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        leaderboard.to_json(f"arena_hard_leaderboard_{date_str}.json", orient="records", indent=4)
