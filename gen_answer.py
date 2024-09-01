"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import re
import time
import concurrent.futures

import tiktoken
import shortuuid
import tqdm

from add_markdown_info import count_markdown_elements, remove_pattern
from utils import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_openai_azure,
    chat_completion_mistral,
    http_completion_gemini,
    chat_completion_cohere,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)


def get_answer(
    question: dict, model: str, endpoint_info: dict, num_choices: int, max_tokens: int, temperature: float, answer_file: str, api_dict: dict
):
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    api_type = endpoint_info["api_type"]

    conv = []

    if "system_prompt" in endpoint_info.keys():
        conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
    elif model in OPENAI_MODEL_LIST:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    for i in range(num_choices):
        turns = []
        for j in range(len(question["turns"])):
            conv.append({"role": "user", "content": question["turns"][j]["content"]})
            if api_type == "anthropic":
                output = chat_completion_anthropic(model=endpoint_info["model_name"],
                                                   messages=conv,
                                                   temperature=temperature,
                                                   max_tokens=max_tokens)
            elif api_type == "mistral":
                output = chat_completion_mistral(model=endpoint_info["model_name"],
                                                 messages=conv,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)
            elif api_type == "gemini":
                output = http_completion_gemini(model=endpoint_info["model_name"],
                                                message=question["turns"][j]["content"],
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            elif api_type == "azure":
                output = chat_completion_openai_azure(model=endpoint_info["model_name"],
                                                      messages=conv,
                                                      temperature=temperature,
                                                      max_tokens=max_tokens,
                                                      api_dict=api_dict)
            elif api_type == "cohere":
                output = chat_completion_cohere(model=endpoint_info["model_name"],
                                                messages=conv,
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            else:
                output = chat_completion_openai(model=endpoint_info["model_name"], 
                                                messages=conv, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                api_dict=api_dict)
            conv.append({"role": "assistant", "content": output})

            turns.append({"content": output})
        choices.append({"index": i, "turns": turns})
    
    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }
    
    if len(choices) == len(turns) == 1:
        metadata = {"token_len": len(encoding.encode(output, 
                                                     disallowed_special=()))}
        ans["conv_metadata"] = metadata | count_markdown_elements(remove_pattern(output, 
                                                                     re.compile("```([^`]*)```")),
                                                                 suffix="")

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    args = parser.parse_args()

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    existing_answer = load_model_answers(os.path.join("data", settings["bench_name"], "model_answer"))
    
    print(settings)

    for model in settings["model_list"]:
        assert model in endpoint_list
        endpoint_info = endpoint_list[model]

        question_file = os.path.join("data", settings["bench_name"], "question.jsonl")
        questions = load_questions(question_file)

        answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model}.jsonl")
        print(f"Output to {answer_file}")

        if "parallel" in endpoint_info:
            parallel = endpoint_info["parallel"]
        else:
            parallel = 1

        # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
        if "tokenizer" in endpoint_info:
            question_list = [question["turns"][0]["content"] for question in questions]
            if model in OPENAI_MODEL_LIST:
                tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
                tokens = [tokenizer.encode(prompt) for prompt in question_list]
                max_tokens = [(settings["max_tokens"] - len(token) - 100) for token in tokens]
            else:
                from transformers import AutoTokenizer
                
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])

                tokens = tokenizer(question_list)
                max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
        else:
            max_tokens = [settings["max_tokens"]] * len(questions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            count = 0
            for index, question in enumerate(questions):
                if model in existing_answer and question["question_id"] in existing_answer[model]:
                    count += 1
                    continue
                future = executor.submit(
                    get_answer,
                    question,
                    model,
                    endpoint_info,
                    settings["num_choices"],
                    max_tokens[index],
                    settings["temperature"],
                    answer_file,
                    get_endpoint(endpoint_info["endpoints"]),
                )
                futures.append(future)
            if count > 0:
                print(f"{count} number of existing answers")
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

        reorg_answer_file(answer_file)
