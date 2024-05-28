import pandas as pd
import numpy as np
import plotly.express as px

import tiktoken
import datetime
import argparse
import os
import math

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from utils import load_model_answers

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if "gpt-4-0314" in models.index:
        elo_scores += 1000 - elo_scores[models["gpt-4-0314"]]
    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=["Model", column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline="gpt-4-0314"):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(judge_name, first_game_only=False, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()
    
    print("Turning judgment results into battles...")

    directory = f"data/arena-hard-v0.1/model_judgment/{judge_name}"
    assert os.path.exists(directory)
    for file in tqdm(glob(f"{directory}/*jsonl")):
        df = pd.read_json(file, lines=True)

        for _, row in df.iterrows():
            # game 1
            output = {"question_id": row["question_id"],
                    "model_a": "gpt-4-0314",
                    "model_b": row["model"]}

            game = row["games"][0]

            weight = 1
            if game["score"] == "A=B":
                output["winner"] = "tie"
            elif game["score"] == "A>B":
                output["winner"] = "model_a"
            elif game["score"] == "A>>B":
                output["winner"] = "model_a"
                weight = WEIGHT
            elif game["score"] == "B>A":
                output["winner"] = "model_b"
            elif game["score"] == "B>>A":
                output["winner"] = "model_b"
                weight = WEIGHT
            else:
                weight = 0

            if weight:
                arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

            if not first_game_only:
                # game 2
                output = {"question_id": row["question_id"],
                        "model_a": "gpt-4-0314",
                        "model_b": row["model"]}

                game = row["games"][1]

                weight = 1
                if game["score"] == "A=B":
                    output["winner"] = "tie"
                elif game["score"] == "A>B":
                    output["winner"] = "model_a"
                elif game["score"] == "A>>B":
                    output["winner"] = "model_a"
                    weight = WEIGHT
                elif game["score"] == "B>A":
                    output["winner"] = "model_b"
                elif game["score"] == "B>>A":
                    output["winner"] = "model_b"
                    weight = WEIGHT
                else:
                    weight = 0

                if weight:
                    arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    arena_hard_battles.to_json("data/arena_hard_battles.jsonl", lines=True, orient="records")
    return arena_hard_battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--baseline", type=str, default="gpt-4-0314")
    parser.add_argument("--load-battles", action="store_true")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")
    args = parser.parse_args()
    print(args)
    assert not args.load_bootstrap or (args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."

    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    model_answers = load_model_answers(answer_dir)
    
    if args.load_battles:
        assert os.path.exists("data/arena_hard_battles.jsonl")
        battles = pd.read_json("data/arena_hard_battles.jsonl", lines=True)
    else:
        battles = get_battles_from_judgment(args.judge_name, args.first_game_only, args.weight)
        
    bootstrap_online_elo = compute_mle_elo(battles)


    if args.load_bootstrap:
        bootstrap_elo_lu = pd.read_json("data/bootstrapping_results.jsonl", lines=True)
    else:
        np.random.seed(42)
        bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, args.num_rounds)
        bootstrap_elo_lu.to_json("data/bootstrapping_results.jsonl", lines=True, orient="records")

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                length += turn["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()
    
    if not args.show_elo:
        stats.sort_values(by="model", inplace=True)
        stats["score"] = get_win_rate_column(stats, "score", args.baseline).tolist()
        stats["lower"] = get_win_rate_column(stats, "lower", args.baseline).tolist()
        stats["upper"] = get_win_rate_column(stats, "upper", args.baseline).tolist()
        decimal = 1
    else:
        decimal = 0
        stats = stats.astype({"score" : int, "lower" : int, "upper" : int})
    
    stats.sort_values(by="score", ascending=False, inplace=True)
    for _, row in stats.iterrows():
        interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
        print(f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}")

    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        stats.to_json(f"arena_hard_leaderboard_{date_str}.json", orient="records", indent=4)
