import json
import shortuuid
import yaml
import argparse
import os
import re
import openai
import random
import concurrent.futures
from tqdm import tqdm
from string import Formatter
from fastchat.llm_judge.common import (
    load_questions,
    chat_compeletion_openai,
    chat_compeletion_openai_azure,
    chat_compeletion_anthropic,
    chat_compeletion_palm,
    load_questions,
    load_model_answers,
)
from fastchat.model.model_adapter import get_conversation_template


pattern = re.compile(r"\[\[(\d*?)\]\]")


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_list=None):
    api_dict = get_endpoint(endpoint_list)

    if model in ["claude-v1", "claude-instant-v1", "claude-2"]:
        output = chat_compeletion_anthropic(model, conv, temperature, max_tokens)
    elif model == "palm-2-chat-bison-001":
        chat_state, output = chat_compeletion_palm(
            chat_state, model, conv, temperature, max_tokens
        )
    elif "azure" in model:
        output = chat_compeletion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_compeletion_openai(model, conv, temperature, max_tokens, api_dict)
    return output


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


# question: dict, answer: dict, reference: dict, configs: dict, output_file: str
def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    configs = args["configs"]
    output_file = args["output_file"]

    model = configs["judge_model"]
    prompt_args = {}

    for i, turn in enumerate(question["turns"]):
        prompt_args[f"question_{i+1}"] = turn

    if answer:
        for i, turn in enumerate(answer["choices"][0]["turns"]):
            prompt_args[f"answer_{i+1}"] = turn
    if reference:
        for i, turn in enumerate(reference["choices"][0]["turns"]):
            prompt_args[f"ref_answer_{i+1}"] = turn

    user_prompt = configs["prompt_template"].format(**prompt_args)

    conv = get_conversation_template(model)
    conv.set_system_message(configs["system_prompt"])

    # add few shot prompts
    if configs["few_shot"]:
        for i, shot in enumerate(configs["few_shot"]):
            conv.append_message(conv.roles[i % 2], shot)

    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    judgment = get_answer(
        model,
        conv,
        configs["temperature"],
        configs["max_tokens"],
        configs["endpoint_list"],
    )

    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(matches) == 1:
        score = int(matches[0])
    else:
        score = None

    with open(output_file, "a") as f:
        output = {
            "question_id":question["question_id"],
            # "answer_id": shortuuid.uuid(),
            "model":answer["model_id"],
            # "prompt": conv.messages,
            "user_prompt":conv.messages[0][1],
            # "judgment": [{"index": 0, "turns": [judgment]}],
            "judgment":judgment,
            "score":score
        }

        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="data/arena-bench-v1_config.yaml")
    parser.add_argument("--bench-name", type=str, default="arena-bench-v1")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify a model's response to be judged.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    print(args)

    configs = make_config(args.config_file)

    question_file = os.path.join("data", args.bench_name, "question.jsonl")
    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    ref_answer_dir = os.path.join("data", args.bench_name, "reference_answer")

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(answer_dir)

    if args.model:
        model_answers = {args.model:model_answers[args.model]}
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = ref_answers[configs["ref_model"]]

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
            "data",
            args.bench_name,
            "model_judgment",
            f"{configs['name']}_judge_any.jsonl",
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for model in model_answers.keys():
            for question in questions:
                kwargs = {}
                kwargs["question"] = question
                kwargs["answer"] = model_answers[model][question["question_id"]]
                if ref_answers:
                    kwargs["reference"] = ref_answers[question["question_id"]]
                else:
                    kwargs["reference"] = None

                kwargs["configs"] = configs
                kwargs["output_file"] = output_file
                future = executor.submit(judgment, **kwargs)
                futures.append(future)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()