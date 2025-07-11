import argparse
import json
import pandas as pd
import os
import time
import concurrent.futures
import tqdm
import yaml
import random
import threading
import orjson
from typing import List, Dict

from category import Category


LOCK = threading.RLock()

TASKS = None
CACHE_DICT = None
OUTPUT_DICT = None

# API setting constants
API_MAX_RETRY = None
API_RETRY_SLEEP = None
API_ERROR_OUTPUT = None


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)
    return config_kwargs


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # extra_body={"guided_choice": GUIDED_CHOICES} if GUIDED_CHOICES else None,
            )
            output = completion.choices[0].message.content
            # print(output)
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
            break
        except openai.APIConnectionError as e:
            print(messages)
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.InternalServerError as e:
            print(messages)
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def get_answer(
    question: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    answer_file: str,
    api_dict: dict,
    categories: list,
    api_type: str,
    testing: bool,
):
    if "category_tag" in question:
        category_tag = question["category_tag"]
    else:
        category_tag = {}

    output_log = {}

    for category in categories:
        conv = category.pre_process(question["prompt"])
        if api_type == "openai":
            output = chat_completion_openai(
                model=model_name,
                messages=conv,
                temperature=temperature,
                max_tokens=max_tokens,
                api_dict=api_dict,
            )
        elif api_type == "anthropic":
            output = chat_completion_anthropic(
                model=model_name,
                messages=conv,
                temperature=temperature,
                max_tokens=max_tokens,
                api_dict=api_dict,
            )
        # Dump answers
        category_tag[category.name_tag] = category.post_process(output)

        if testing:
            output_log[category.name_tag] = {"output": output,
                                             "conversation": conv}

    question["category_tag"] = category_tag
    if testing:
        question["output_log"] = output_log

    question.drop(["prompt", "uid", "required_tasks"], inplace=True)

    with LOCK:
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(question.to_dict()) + "\n")


def category_merge(row):
    id = row["uid"]
    input_category = row["category_tag"] if "category_tag" in row else {}
    cache_category = CACHE_DICT[id]["category_tag"] if id in CACHE_DICT else {}
    output_category = OUTPUT_DICT[id]["category_tag"] if id in OUTPUT_DICT else {}

    # tries to fill in missing categories using cache first, then output
    for name in TASKS:
        if name not in input_category:
            if name in cache_category:
                input_category[name] = cache_category[name]
                continue
            if name in output_category:
                input_category[name] = output_category[name]

    return input_category


def find_required_tasks(row):
    id = row["uid"]
    input_category = row["category_tag"] if "category_tag" in row else {}
    cache_category = CACHE_DICT[id]["category_tag"] if id in CACHE_DICT else {}
    output_category = OUTPUT_DICT[id]["category_tag"] if id in OUTPUT_DICT else {}

    return [
        name
        for name in TASKS
        if not (
            name in input_category or name in cache_category or name in output_category
        )
    ]

def _get_prompt(convo: List[Dict]):
    prompt = ""
    for i in range(0, len(convo), 2):
        if isinstance(convo[i]['content'], str):
            prompt += f"{convo[i]['content']}\n"
        else:
            prompt += f"{convo[i]['content'][0]}\n"
    return prompt

def _get_uid(row: pd.Series):
    if "question_id" in row.index and "tstamp" in row.index:
        return str(row["question_id"]) + str(row["tstamp"])
    else:
        return str(row.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    enter = input(
        "Make sure your config file is properly configured. Press enter to continue."
    )
    if not enter == "":
        exit()

    config = make_config(args.config)

    API_MAX_RETRY = config["max_retry"]
    API_RETRY_SLEEP = config["retry_sleep"]
    API_ERROR_OUTPUT = config["error_output"]

    categories = [Category.create_category(name) for name in config["task_name"]]
    TASKS = config["task_name"]
    print(
        f"Following categories will be labeled:\n{[category.name_tag for category in categories]}"
    )

    print("loading input data (might take min)")
    with open(config["input_file"], "rb") as f:
        data = orjson.loads(f.read())
    input_data = pd.DataFrame(data)

    # much faster than pd.apply
    input_data["uid"] = input_data.apply(_get_uid, axis=1)
    assert len(input_data) == len(input_data.uid.unique())
    print(f"{len(input_data)}# of input data just loaded")

    if config["cache_file"]:
        print("loading cache data")
        with open(config["cache_file"], "rb") as f:
            data = orjson.loads(f.read())
        cache_data = pd.DataFrame(data)
        cache_data["uid"] = cache_data.question_id.map(str) + cache_data.tstamp.map(str)
        assert len(cache_data) == len(cache_data.uid.unique())

        print(f"{len(cache_data)}# of cache data just loaded")

        assert "category_tag" in cache_data.columns
        cache_dict = cache_data[["uid", "category_tag"]].set_index("uid")
        print("finalizing cache_dict (should take less than 30 sec)")
        CACHE_DICT = cache_dict.to_dict("index")
    else:
        CACHE_DICT = {}

    if os.path.isfile(config["output_file"]):
        print("loading existing output")
        output_data = pd.read_json(config["output_file"], lines=True)
        output_data["uid"] = output_data.apply(_get_uid, axis=1)
        assert len(output_data) == len(output_data.uid.unique())

        print(f"{len(output_data)}# of existing output just loaded")

        assert "category_tag" in output_data.columns
        output_dict = output_data[["uid", "category_tag"]].set_index("uid")
        print("finalizing output_dict (should take less than 30 sec)")
        OUTPUT_DICT = output_dict.to_dict("index")
    else:
        OUTPUT_DICT = {}

    print(
        "finding tasks needed to run... (should take around 1 minute or less on large dataset)"
    )
    input_data["required_tasks"] = input_data.apply(find_required_tasks, axis=1)

    not_labeled = input_data[input_data.required_tasks.map(lambda x: len(x) > 0)].copy()

    print(f"{len(not_labeled)} # of conversations needs to be labeled")
    for name in TASKS:
        print(
            f"{name}: {len(not_labeled[not_labeled.required_tasks.map(lambda tasks: name in tasks)])}"
        )

    not_labeled["prompt"] = not_labeled.conversation_a.map(_get_prompt)
    not_labeled["prompt"] = not_labeled.prompt.map(lambda x: x[:12500])

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["parallel"]
    ) as executor:
        futures = []
        for index, row in tqdm.tqdm(not_labeled.iterrows()):
            future = executor.submit(
                get_answer,
                row,
                config["model_name"],
                config["max_token"],
                config["temperature"],
                config["output_file"],
                get_endpoint(config["endpoints"]),
                [
                    category
                    for category in categories
                    if category.name_tag in row["required_tasks"]
                ],
                config["api_type"],
                args.testing,
            )
            futures.append(future)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    if config["convert_to_json"]:
        # merge two data frames, but only take the fields from the cache data to overwrite the input data
        merge_columns = [category.name_tag for category in categories]
        print(f"Columns to be merged:\n{merge_columns}")

        input_data["uid"] = input_data.apply(_get_uid, axis=1)
        assert len(input_data) == len(input_data.uid.unique())

        # fastest way to merge
        assert os.path.isfile(config["output_file"])
        print("reading output file...")
        temp = pd.read_json(config["output_file"], lines=True)
        temp["uid"] = temp.apply(_get_uid, axis=1)
        assert len(temp) == len(temp.uid.unique())

        assert "category_tag" in temp.columns
        output_dict = temp[["uid", "category_tag"]].set_index("uid")
        print("finalizing output_dict (should take less than 30 sec)")
        OUTPUT_DICT = output_dict.to_dict("index")

        print("begin merging (should take around 1 minute or less on large dataset)")
        input_data["category_tag"] = input_data.apply(category_merge, axis=1)
        print("merge completed")

        final_data = input_data.drop(
            columns=["prompt", "uid", "required_tasks"], errors="ignore"
        )
        final_data.to_json(
            config["output_file"][:-1], orient="records", indent=4, force_ascii=False
        )
