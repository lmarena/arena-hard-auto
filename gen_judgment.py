import json
import yaml
import argparse
import os
import re
import concurrent.futures

from tqdm import tqdm

from utils import (
    load_questions,
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    return output


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id":question["question_id"],
        "model":answer["model_id"],
        "judge": model,
        "games":[]
        }

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    temp = baseline
                    baseline = answer
                    answer = temp

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(2):
            new_judgment = get_answer(
                model,
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score":score
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:
                    kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
