import pandas as pd
import numpy as np
import math
import inspect

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

np.random.seed(42)

STYLE_CONTROL_ELEMENTS = [
    "sum_assistant_a_tokens",
    "header_count_a",
    "list_count_a",
    "bold_count_a",
    "sum_assistant_b_tokens",
    "header_count_b",
    "list_count_b",
    "bold_count_b",
]

LENGTH_CONTROL_ELEMENTS = [
    "sum_assistant_a_tokens",
    "sum_assistant_b_tokens",
]

MARKDOWN_CONTROL_ELEMENTS = [
    "header_count_a",
    "list_count_a",
    "bold_count_a",
    "header_count_b",
    "list_count_b",
    "bold_count_b",
]


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, baseline_model="gpt-4-0314"):
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
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model="gpt-4-0314"):
    rows = []
    kwargs = {}
    if baseline_model in inspect.signature(func_compute_elo).parameters:
        kwargs[baseline_model] = baseline_model
    for _ in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs))
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


def fit_bt(X, Y, models, SCALE=400, INIT_RATING=1000, baseline_model="gpt-4-0314"):
    from sklearn.linear_model import LogisticRegression

    p = len(models.index)

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-13b to 800 if applicable
    assert baseline_model in models.index
    
    elo_scores += 1114 - elo_scores[models[baseline_model]]
    return (
        pd.Series(elo_scores[:p], index=models.index).sort_values(ascending=False),
        lr.coef_[0][p:],
    )
    
    
def construct_style_matrices(
    df,
    BASE=10,
    apply_ratio=[1, 1, 1, 1],
    style_elements=STYLE_CONTROL_ELEMENTS,
    add_one=True,
):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]
    assert len(style_elements) % 2 == 0
    k = int(len(style_elements) / 2)

    X = np.zeros([n, p + k])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # creates turn each of the specified column in "conv_metadata" into a vector
    style_vector = np.array(
        [
            df.conv_metadata.map(
                lambda x: x[element]
                if type(x[element]) is int
                else sum(x[element].values())
            ).tolist()
            for element in style_elements
        ]
    )

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    style_diff[apply_ratio] /= style_sum[
        apply_ratio
    ]  # Apply ratio where necessary (length, etc)

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)

    X[:, -k:] = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    return X, Y, models


def get_bootstrap_result_style_control(X, Y, battles, models, func_compute_elo, num_round=1000, baseline_model="gpt-4-0314"):
    elos = []
    coefs = []
    assert X.shape[0] % 2 == 0 and X.shape[0] == Y.shape[0]
    k = int(
        X.shape[0] / 2
    )  # Since we duplicate the battles when constructing X and Y, we don't want to sample the duplicates

    battles_tie_idx = (battles["winner"] == "tie") | (battles["winner"] == "tie (bothbad)")
    for _ in tqdm(range(num_round), desc="bootstrap"):
        indices = np.random.choice(list(range(k)), size=(k), replace=True)

        index2tie = np.zeros(k, dtype=bool)
        index2tie[battles_tie_idx] = True

        nontie_indices = indices[~index2tie[indices]]
        tie_indices = np.concatenate([indices[index2tie[indices]], indices[index2tie[indices]]+k])

        _X = np.concatenate([X[nontie_indices], X[nontie_indices], X[tie_indices]])
        _Y = np.concatenate([Y[nontie_indices], Y[nontie_indices], Y[tie_indices]])

        assert _X.shape == X.shape and _Y.shape == Y.shape

        states = ~_X[:, : len(models)].any(axis=0)

        elo, coef = func_compute_elo(_X, _Y, models[~states], baseline_model=baseline_model)
        elos.append(elo)
        coefs.append(coef)

    df = pd.DataFrame(elos)
    return df[df.median().sort_values(ascending=False).index], coefs