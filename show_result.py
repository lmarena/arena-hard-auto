import pandas as pd
import argparse
import os
import torch
from glob import glob
from tqdm import tqdm

from utils.judge_utils import JUDGE_SETTINGS
from utils.math_utils import one_hot_encode, to_winrate_probabilities, bootstrap_pairwise_model


STYLE_FEATURES_MAP = {
    "markdown": [
        "header_count",
        "list_count",
        "bold_count",
    ],
    "length": [
        "token_len",
    ]
}


def load_judgments(judge_name, benchmark, weight=3):
    data = pd.concat([
        pd.read_json(f, lines=True) for f in tqdm(glob(os.path.join(
            "data",
            benchmark, 
            "model_judgment", 
            judge_name, 
            "*.jsonl"
        )))
    ]).reset_index(drop=True)

    _data = data[~data.games.map(lambda x: x[0] is None or x[1] is None)]
    _data = _data[~_data.games.map(lambda x: x[0]['score'] is None or x[1]['score'] is None)].reset_index(drop=True)
    
    print(f"Number of null judgments found: {len(data) - len(_data)}")
    
    # map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )
    
    battles = _data[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
    
    return battles


def get_model_style_metadata(benchmark):
    model_metadata = {}
    for file in glob(os.path.join("data", benchmark, "model_answer", "*.jsonl")):
        df = pd.read_json(file, lines=True)
        model_metadata[df.iloc[0]['model']] = df.set_index('uid')['metadata'].to_dict()
        
    return model_metadata


def print_leaderboard(battles):
    for cat in battles.category.unique():
        baseline = JUDGE_SETTINGS[cat]["baseline"]
        
        _battles = battles[battles.category == cat].drop(columns=['category'])[['model', 'scores']]
        
        # remove model path
        _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
        
        leaderboard = _battles.groupby("model").mean().reset_index()
        
        leaderboard = pd.concat(
            [leaderboard, pd.DataFrame({"model": baseline, "scores": 0.5}, index=[0])]
        ).sort_values(by="scores", ascending=False).reset_index(drop=True)
        
        print(f"##### Category: {cat} #####")
        print(leaderboard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-name", type=str, default="gemini-2.5")
    parser.add_argument("--benchmark", type=str, default="arena-hard-v2.0")
    parser.add_argument("--control-features", nargs="+", default=[])
    args = parser.parse_args()
    
    if args.control_features:
        control_features = []
        for feature in args.control_features:
            assert feature in STYLE_FEATURES_MAP, f"Feature {feature} not found in {STYLE_FEATURES_MAP}"
            control_features.extend(STYLE_FEATURES_MAP[feature])
            
        print(f"INFO: Control features: {control_features}")
        
        battles = load_judgments(args.judge_name, args.benchmark)
        
        style_metadata = get_model_style_metadata(args.benchmark)
        
        battles['model_style'] = battles.apply(lambda row: 
            style_metadata[row['model']][row['uid']], 
            axis=1
        )
        battles['baseline_style'] = battles.apply(
            lambda row: style_metadata[JUDGE_SETTINGS[row['category']]["baseline"]][row['uid']], 
            axis=1
        )
        
        style_features_a = battles.model_style.map(lambda x:
            torch.tensor([v if isinstance(v, int) else sum(v.values()) for k, v in x.items() if k in control_features])
        )
        style_features_b = battles.baseline_style.map(lambda x:
            torch.tensor([v if isinstance(v, int) else sum(v.values()) for k, v in x.items() if k in control_features])
        )
        
        battles['style_feature_diff'] = [
            style_features_a[i] - style_features_b[i] for i in range(len(style_features_a))
        ]
        
        for cat in battles.category.unique():
            _battles = battles[battles.category == cat].reset_index(drop=True)
            
            if cat == "hard_prompt":
                import pickle
                with open("tmp_battles.pkl", "wb") as f:
                    pickle.dump(_battles, f)
                print("file saved")
        
            outcomes = torch.tensor(_battles.scores.tolist())
            model_features, unique_models = one_hot_encode(_battles.model.tolist())

            style_feature_diff = torch.tensor([diff.tolist() for diff in _battles.style_feature_diff])
            
            features = torch.cat([model_features, style_feature_diff], dim=1)
            
            print(features.shape)

            coefs, _ = bootstrap_pairwise_model(features, outcomes, loss_type="bt")
            
            _coefs = coefs[:, :-len(control_features)]
            
            table = pd.DataFrame(
                columns=unique_models, 
                data=to_winrate_probabilities(
                    _coefs, 
                    unique_models
                ).tolist()
            )
            
            print(f"##### Category: {cat} #####")
            print(table.quantile(0.5).sort_values(ascending=False))
    else:
        battles = load_judgments(args.judge_name, args.benchmark)
        print_leaderboard(battles)
        