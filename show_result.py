import pandas as pd
import numpy as np

import datetime
import argparse
import os

from glob import glob
from tqdm import tqdm

from utils import load_model_answers
from utils_math import (
    compute_mle_elo, 
    get_bootstrap_result,
    get_win_rate_column,
    fit_bt,
    construct_style_matrices,
    get_bootstrap_result_style_control,
    STYLE_CONTROL_ELEMENTS,
    LENGTH_CONTROL_ELEMENTS,
    MARKDOWN_CONTROL_ELEMENTS,
)


def get_battles_from_row(row, first_game_only, multiplier, baseline_model, metadata=None):
    results = []
    output = {"question_id": row["question_id"],
              "model_a": baseline_model,
              "model_b": row["model"]}
    
    game = row["games"][0]
    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_a"
    elif game["score"] == "A>>B":
        output["winner"] = "model_a"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_b"
    elif game["score"] == "B>>A":
        output["winner"] = "model_b"
        weight = multiplier
    else:
        weight = 0
    
    # add conv_metadata for style control
    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]]["conv_metadata"]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]]["conv_metadata"]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["list_count"],
            "bold_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["bold_count"],
        }

    if weight:
        results += [output] * weight
        
    if first_game_only:
        return results
    
    # game 2
    output = {"question_id": row["question_id"],
            "model_a": baseline_model,
            "model_b": row["model"]}

    game = row["games"][1]

    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_b"
    elif game["score"] == "A>>B":
        output["winner"] = "model_b"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_a"
    elif game["score"] == "B>>A":
        output["winner"] = "model_a"
        weight = multiplier
    else:
        weight = 0
    
    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]]["conv_metadata"]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]]["conv_metadata"]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["list_count"],
            "bold_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["bold_count"],
        }

    if weight:
        results += [output] * weight
    
    return results


def get_battles_from_judgment(bench_name, 
                              judge_name, 
                              first_game_only=False, 
                              multiplier=3, 
                              baseline_model="gpt-4-0314",
                              style_control=False):
    print("Turning judgment results into battles...")

    judge_dir = f"data/{bench_name}/model_judgment/{judge_name}"
    assert os.path.exists(judge_dir)
    judgments = pd.concat([pd.read_json(file, lines=True) for file in tqdm(glob(f"{judge_dir}/*jsonl"))])
    
    metadata = None
    if style_control:
        ans_dir = f"data/{bench_name}/model_answer"
        assert os.path.exists(ans_dir)
        
        metadata = {}
        for file in tqdm(glob(f"{ans_dir}/*.jsonl")):
            df = pd.read_json(file, lines=True)
            assert "conv_metadata" in df.columns, "You must have conv_metadata attributes in your model answer to apply style contro. Please pull newest data if needed."
            metadata[df.model_id[0]] = df[["question_id", "conv_metadata"]].set_index("question_id").to_dict("index")
    
    battles = judgments.apply(lambda row: get_battles_from_row(row, first_game_only, multiplier, baseline_model, metadata), axis=1)
    battles = pd.DataFrame(battles[battles.map(len) > 0].explode().tolist())
    battles.to_json("data/arena_hard_battles.jsonl", orient="records", lines=True)
    return battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--baseline", type=str, default="gpt-4-0314")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")
    parser.add_argument("--style-control", action="store_true")
    parser.add_argument("--length-control-only", action="store_true")
    parser.add_argument("--markdown-control-only", action="store_true")
    args = parser.parse_args()
    print(args)
    assert not args.load_bootstrap or (args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."
    assert sum([args.style_control, args.length_control_only, args.markdown_control_only]) < 2, "You can only control one of the three: length, markdown, or both style."

    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    model_answers = load_model_answers(answer_dir)
    
    battles = get_battles_from_judgment(args.bench_name, 
                                        args.judge_name, 
                                        args.first_game_only, 
                                        args.weight, 
                                        args.baseline,
                                        args.style_control or args.length_control_only or args.markdown_control_only)
    
    if args.style_control:
        X, Y, models = construct_style_matrices(battles)
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=args.baseline)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(X, Y, battles, models, 
                                                                     fit_bt, 
                                                                     num_round=args.num_rounds, 
                                                                     baseline_model=args.baseline)
        display_coefs = {STYLE_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(STYLE_CONTROL_ELEMENTS) // 2)}
        print(f"Style Coefficients: {display_coefs}")
    elif args.length_control_only:
        X, Y, models = construct_style_matrices(battles, 
                                                apply_ratio=[1], 
                                                style_elements=LENGTH_CONTROL_ELEMENTS)
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=args.baseline)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(X, Y, battles, models, 
                                                                     fit_bt, 
                                                                     num_round=args.num_rounds, 
                                                                     baseline_model=args.baseline)
        display_coefs = {LENGTH_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(LENGTH_CONTROL_ELEMENTS) // 2)}
        print(f"Style Coefficients: {display_coefs}")
    elif args.markdown_control_only:
        X, Y, models = construct_style_matrices(battles, 
                                                apply_ratio=[1, 1, 1], 
                                                style_elements=MARKDOWN_CONTROL_ELEMENTS)
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=args.baseline)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(X, Y, battles, models, 
                                                                     fit_bt, 
                                                                     num_round=args.num_rounds, 
                                                                     baseline_model=args.baseline)
        display_coefs = {MARKDOWN_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(MARKDOWN_CONTROL_ELEMENTS) // 2)}
        print(f"Style Coefficients: {display_coefs}")
    else:
        bt_model_coef = compute_mle_elo(battles, baseline_model=args.baseline)
        bootstrap_model_coef = get_bootstrap_result(battles, compute_mle_elo, args.num_rounds, args.baseline)

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bt_model_coef.index):
        assert model in bootstrap_model_coef.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bt_model_coef[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_model_coef[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_model_coef[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                if "token_len" in turn:
                    length += turn["token_len"]
                else:
                    length += row["conv_metadata"]["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_model_coef[model].tolist()
    
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

    # If outputting leaderboard to a csv file.
    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        stats = stats.drop(columns=['results'])
        CI = []
        for i in range(len(stats)):
            score = stats.iloc[i]['score']
            upper = stats.iloc[i]['upper']
            lower = stats.iloc[i]['lower']
            CI.append(f"(-{(score-lower):.2f}, +{(upper-score):.2f})")

        stats["CI"] = CI
        col_list = list(stats)
        stats = stats.loc[:,col_list]
        stats.rename(columns={'upper': 'rating_q975'}, inplace=True)
        stats.rename(columns={'lower': 'rating_q025'}, inplace=True)

        col_list = list(stats)
        col_list[-2], col_list[-1] = col_list[-1], col_list[-2]
        stats = stats.loc[:,col_list]
        stats['date'] = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
        stats.to_csv(f"leaderboard/arena_hard_leaderboard_{date_str}.csv", index=False)