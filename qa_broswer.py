import argparse
import json
from collections import defaultdict
import re
import glob
import os
import yaml

import gradio as gr

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
)


questions = []
model_answers = {}
baseline_model = None

model_judgments_normal_single = {}
model_judgments_math_single = {}

model_judgments_normal_pairwise = {}
model_judgments_math_pairwise = {}

question_selector_map = {}
category_selector_map = defaultdict(list)

model_reference_answers = []


def display_question(category_selector, request: gr.Request):
    choices = category_selector_map['arena-bench-v1']
    return gr.Dropdown.update(
        value=choices[0],
        choices=choices,
    )


def display_pairwise_answer(
    question_selector, model_selector1, model_selector2, request: gr.Request
):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    ans2 = model_answers[model_selector2][qid]

    if baseline_model:
        ans3 = model_answers[baseline_model][qid]
    else:
        ans3 = model_judgments_normal_single

    chat_mds = pairwise_to_gradio_chat_mds(q, ans1, ans2, ans_base=ans3)
    gamekey = (qid, model_selector1, model_selector2)

    judgment_dict = model_judgments_math_pairwise[qid]

    explanations = get_pairwise_judge_explanation(gamekey, judgment_dict)
    return chat_mds + explanations


def display_single_answer(question_selector, model_selector1, request: gr.Request):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    
    chat_mds = single_to_gradio_chat_mds(q, ans1)
    gamekey = (qid, model_selector1)

    judgment_dict = model_judgments_math_single[("gpt-4", "single-math-v1")]

    explanation = "##### Model Judgment (first turn)\n" + get_single_judge_explanation(
        gamekey, judgment_dict
    )

    return chat_mds + [explanation]


newline_pattern1 = re.compile("\n\n(\d+\. )")
newline_pattern2 = re.compile("\n\n(- )")


def post_process_answer(x):
    """Fix Markdown rendering problems."""
    x = x.replace("\u2022", "- ")
    x = re.sub(newline_pattern1, "\n\g<1>", x)
    x = re.sub(newline_pattern2, "\n\g<1>", x)
    return x


def pairwise_to_gradio_chat_mds(question, ans_a, ans_b, ans_base=None, turn=None):
    end = len(question["turns"]) if turn is None else turn + 1
    size = end * 3

    mds = ["" for i in range(size + len(model_reference_answers))]
    for i in range(end):
        base = i * 3
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        mds[base + 1] = "##### Assistant A\n" + post_process_answer(
            ans_a["choices"][0]["turns"][i].strip()
        )
        mds[base + 2] = "##### Assistant B\n" + post_process_answer(
            ans_b["choices"][0]["turns"][i].strip()
        )

    if ans_base:
        mds += [""]
        mds[size] = f"##### Baseline\n" + post_process_answer(
            ans_base["choices"][0]["turns"][0].strip()
        )
        size += 1

    for i, model_references in enumerate(model_reference_answers):
        model, references = model_references
        ref = references[question["question_id"]]

        mds[size + i] = f"##### Reference Solution: {model}\n{ref}"
    return mds


def single_to_gradio_chat_mds(question, ans, turn=None):
    end = len(question["turns"]) if turn is None else turn + 1

    mds = ["" for i in range(2 + len(model_reference_answers))]
    for i in range(end):
        base = i * 2
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        mds[base + 1] = "##### Assistant A\n" + post_process_answer(
            ans["choices"][0]["turns"][i].strip()
        )

    for i, model_references in enumerate(model_reference_answers):
        model, references = model_references
        ref = references[question["question_id"]]

        mds[2 + i] = f"##### Reference Solution: {model}\n{ref}"
    return mds


def build_question_selector_map():
    global question_selector_map, category_selector_map

    # Build question selector map
    for i, q in enumerate(questions):
        preview = f"{i+1}: " + q["turns"][0][:128] + "..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


def build_pairwise_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 2
    num_turns = 1
    side_names = ["A", "B"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                if i == 0:
                    value = "Starling-LM-7B-alpha"
                    # value = "gpt-3.5-turbo"
                else:
                    value = "vicuna-33b-v1.3"
                    # value = "gpt-3.5-turbo"
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=value,
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    if baseline_model:
        baseline = gr.Markdown()
        chat_mds.append(baseline)
    
    gr.Markdown("## Model Judgment Comparison \n")

    model_explanations = []
    with gr.Row():
        for j in range(num_sides):
            with gr.Column(scale=100):
                model_explanations.append(gr.Markdown(elem_id="model_explanation"))

            if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    for i in range(len(model_reference_answers)):
        reference = gr.Markdown(elem_id=f"reference_{i+1}")
        chat_mds.append(reference)

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_pairwise_answer,
        [question_selector] + model_selectors,
        chat_mds + model_explanations,
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_pairwise_answer,
            [question_selector] + model_selectors,
            chat_mds + model_explanations,
        )

    return (category_selector,)


def build_single_answer_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 1
    num_turns = 1
    side_names = ["A"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, value='arena-bench-v1', label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=models[i] if len(models) > i else "",
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    for i in range(len(model_reference_answers)):
        reference = gr.Markdown(elem_id=f"reference_{i+1}")
        chat_mds.append(reference)

    model_explanation = gr.Markdown(elem_id="model_explanation")

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_single_answer,
        [question_selector] + model_selectors,
        chat_mds + [model_explanation],
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_single_answer,
            [question_selector] + model_selectors,
            chat_mds + [model_explanation],
        )

    return (category_selector,)


block_css = """
#user_question_1 {
    background-color: #0B0F19;
}
#user_question_2 {
    background-color: #0B0F19;
}
#model_explanation {
    background-color: #0B0F19;
}
"""



def load_demo():
    dropdown_update = gr.Dropdown.update(value=list(category_selector_map.keys())[0])
    return dropdown_update, dropdown_update


def build_demo():
    build_question_selector_map()

    with gr.Blocks(
        title="Arena-Bench Browser",
        theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg),
        css=block_css,
    ) as demo:
        gr.Markdown(
            """
# Arena-Bench-V1
The code to generate answers and judgments is at [arena-bench](https://github.com/lm-sys/arena-bench).
"""
        )
        with gr.Tab("Pairwise Comparison"):
            (category_selector2,) = build_pairwise_browser_tab()
        # with gr.Tab("Single Answer Grading"):
        #     (category_selector,) = build_single_answer_browser_tab()
        demo.load(load_demo, [], [category_selector2])

    return demo


def load_reference(directory, ref_models):
    all_references = []
    for model in ref_models:
        model_references = {}
        with open(os.path.join(directory, f"{model}.jsonl")) as fin:
                for line in fin:
                    line = json.loads(line)
                    model_references[line["question_id"]] = line["choices"][0]["turns"][0]
        all_references.append((model, model_references))
    return all_references


# def load_pairwise_model_judgments(dir: str):
#     """Load model judgments.

#     The return value is a dict of type:
#     Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
#     """
#     filenames = glob.glob(os.path.join(dir, "*.jsonl"))
#     filenames.sort()

#     judge_dict = {}
#     for filename in filenames:
#         for line in open(filename):
#             obj = json.loads(line)
#             qid, model = obj["question_id"], obj["model"]

#             if qid not in judge_dict:
#                 judge_dict[qid] = {}

#             judge_dict[qid][model] = obj["judgment"]

#     return judge_dict


def load_pairwise_model_judgments(dir: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    filenames = glob.glob(os.path.join(dir, "*.jsonl"))
    filenames.sort()

    judge_dict = {}
    for filename in filenames:
        for line in open(filename):
            obj = json.loads(line)
            qid, model = obj["question_id"], obj["model"]

            if qid not in judge_dict:
                judge_dict[qid] = {}

            judge_dict[qid][model] = [game["judgment"] for game in obj["games"]]

    return judge_dict


def load_single_model_judgments(dir: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    filenames = glob.glob(os.path.join(dir, "*.jsonl"))
    filenames.sort()

    judge_dict = {}
    for filename in filenames:
        for line in open(filename):
            obj = json.loads(line)
            judge = tuple(["gpt-4","single-math-v1"])
            qid, model = obj["question_id"], obj["model"]

            if judge not in judge_dict:
                judge_dict[judge] = {}
 
            gamekey = (qid, model)

            judge_dict[judge][gamekey] = {
                "score": obj["score"],
                "judgment": obj["judgment"],
            }
    return judge_dict


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model_1, model_2 = gamekey
        
        g1_judgment, g2_judgment = judgment_dict[model_1], judgment_dict[model_2]

        # g1_judgment = g1_judgment[:-6] + f"<mark><span style='color:black'>{g1_judgment[-6:]}</span></mark>"
        # g2_judgment = g2_judgment[:-6] + f"<mark><span style='color:black'>{g2_judgment[-6:]}</span></mark>"
        return [f"**Assistant**: {model_1}\n\n" + f"**<mark><span style='color:black'>Game 1 Judgment</span></mark>**: {g1_judgment[0]}\n\n" + f"**<mark><span style='color:black'>Game 2 Judgment</span></mark>**: {g1_judgment[1]}",
                f"**Assistant**: {model_2}\n\n" + f"**<mark><span style='color:black'>Game 1 Judgment</span></mark>**: {g2_judgment[0]}\n\n" + f"**<mark><span style='color:black'>Game 2 Judgment</span></mark>**: {g2_judgment[1]}"]
    except KeyError:
        return "N/A"
    

def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Assistant**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bench-name", type=str, default="arena-bench-v1")
    parser.add_argument("--judge-mode", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="data/arena-bench-v1_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.config_file)

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    pairwise_model_judgment_dir = (
        os.path.join("data", args.bench_name, args.judge_mode)
    )
    single_model_judgment_dir = (
        os.path.join("data", args.bench_name, args.judge_mode)
    )
    reference_dir = os.path.join("data", args.bench_name, "reference_answer")

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)

    # Load model judgments
    # model_judgments_normal_single = (
    #     model_judgments_math_single
    # ) = load_single_model_judgments(single_model_judgment_dir)
    model_judgments_normal_pairwise = (
        model_judgments_math_pairwise
    ) = load_pairwise_model_judgments(pairwise_model_judgment_dir)

    model_reference_answers = load_reference(reference_dir, configs["ref_model"])

    if configs["baseline"]:
        baseline_model = configs["baseline_model"]

    for i in range(len(model_reference_answers)):
        block_css += f"\n#reference_{i}\n{{\n background-color: #0B0F19;\n}}"

    demo = build_demo()
    demo.queue(concurrency_count=10, status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
