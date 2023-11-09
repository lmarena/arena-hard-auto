import argparse
import json
from collections import defaultdict
import re
import glob
import os

import gradio as gr

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    get_single_judge_explanation,
)


questions = []
model_answers = {}

model_judgments_normal_single = {}
model_judgments_math_single = {}
model_reference_answers = {}

question_selector_map = {}
category_selector_map = defaultdict(list)


def display_question(category_selector, request: gr.Request):
    choices = category_selector_map['arena-bench-v1']
    return gr.Dropdown.update(
        value=choices[0],
        choices=choices,
    )


def display_single_answer(question_selector, model_selector1, request: gr.Request):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    print(model_answers[model_selector1].keys())
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


def single_to_gradio_chat_mds(question, ans, turn=None):
    end = len(question["turns"]) if turn is None else turn + 1

    mds = ["", "", ""]
    for i in range(end):
        base = i * 2
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        mds[base + 1] = "##### Assistant A\n" + post_process_answer(
            ans["choices"][0]["turns"][i].strip()
        )

    ref = model_reference_answers[question["question_id"]]

    mds[2] = f"##### Reference Solution\n{ref}"
    return mds


def build_question_selector_map():
    global question_selector_map, category_selector_map

    # Build question selector map
    for i, q in enumerate(questions):
        preview = f"{i+1}: " + q["turns"][0][:128] + "..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


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

    reference = gr.Markdown(elem_id=f"reference")
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
#reference {
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
The code to generate answers and judgments is at [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
"""
        )
        with gr.Tab("Single Answer Grading"):
            (category_selector,) = build_single_answer_browser_tab()
        # with gr.Tab("Pairwise Comparison"):
        #     (category_selector2,) = build_pairwise_browser_tab()
        demo.load(load_demo, [], category_selector)

    return demo


def load_reference(filename):
    references = {}
    with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                references[line["question_id"]] = line["choices"][0]["turns"][0]
    return references


def load_single_model_judgments(dir: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    filenames = glob.glob(os.path.join(dir, "*.jsonl"))
    filenames.sort()

    judge_dict = {}
    for filename in filenames:
        print(filename)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bench-name", type=str, default="arena-bench-v1")
    args = parser.parse_args()
    print(args)

    question_file = f"data/{args.bench_name}/question.jsonl"
    reference_file = f"data/{args.bench_name}/reference_answer/gpt-4.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    judgment_dir = (
        f"data/{args.bench_name}/model_judgment"
    )

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)

    # Load model judgments
    model_judgments_normal_single = (
        model_judgments_math_single
    ) = load_single_model_judgments(judgment_dir)

    model_reference_answers = load_reference(reference_file)

    demo = build_demo()
    demo.queue(concurrency_count=10, status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )