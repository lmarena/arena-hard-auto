import os
import json
import time
import yaml
import random
import shortuuid
import pandas as pd

from glob import glob
from tqdm import tqdm

# API setting constants
API_MAX_RETRY = 3
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = None

registered_api_completion = {}
registered_engine_completion = {}


def register_api(api_type):
    def decorator(func):
        registered_api_completion[api_type] = func
        return func

    return decorator


def register_engine(engine_type):
    def decorator(func):
        registered_engine_completion[engine_type] = func
        return func

    return decorator


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["uid"]] = line
        model_answers[model_name] = answer

    return model_answers


def load_id_to_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                
                if line["uid"] in model_answers:
                    model_answers[line["uid"]][model_name] = line
                else:
                    model_answers[line["uid"]] = {model_name: line}
                
    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


@register_api("openai")
def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
        
    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                # temperature=temperature,
                max_tokens=max_tokens,
                )
            output = {
                "answer": completion.choices[0].message.content
            }
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


@register_api("openai_thinking")
def chat_completion_openai_thinking(model, messages, api_dict=None, **kwargs):
    import openai
    
    if api_dict:
        client = openai.OpenAI(
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else 'medium',
            )
            output = {
                "answer": completion.choices[0].message.content
            }
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


@register_api("deepseek_reasoner")
def chat_completion_deepseek_reasoner(messages, api_dict, **kwargs):
    import urllib.request

    chat_endpoint_headers = {
        "User-Agent": "curl/8.7.1",
        "Authorization": "Bearer {}".format(api_dict['api_key']),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    chat_endpoint_url = "https://api.deepseek.com/chat/completions"
    
    req_body = {
        "messages": messages,
        "model": "deepseek-reasoner",
        "stream": False,
    }
    req_data = json.dumps(req_body).encode("utf-8")
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            req = urllib.request.Request(
                chat_endpoint_url,
                headers = chat_endpoint_headers.copy(),
                data = req_data,
            )
            
            with urllib.request.urlopen(req) as res:
                res_data = res.read()
            res_body = json.loads(res_data.decode("utf-8"))
            
            output = {
                "thought": res_body["choices"][0]["message"]["reasoning_content"],
                "answer": res_body["choices"][0]["message"]["content"],
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output


@register_api("deepseek")
def chat_completion_deepseek(messages, max_tokens, api_dict, **kwargs):
    import urllib.request

    chat_endpoint_headers = {
        "User-Agent": "curl/8.7.1",
        "Authorization": "Bearer {}".format(api_dict['api_key']),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    chat_endpoint_url = "https://api.deepseek.com/chat/completions"
    
    req_body = {
        "messages": messages,
        "model": "deepseek-chat",
        "stream": False,
        "max_tokens": max_tokens,
    }
    req_data = json.dumps(req_body).encode("utf-8")
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            req = urllib.request.Request(
                chat_endpoint_url,
                headers = chat_endpoint_headers.copy(),
                data = req_data,
            )
            
            with urllib.request.urlopen(req) as res:
                res_data = res.read()
            res_body = json.loads(res_data.decode("utf-8"))
            
            output = {
                "answer": res_body["choices"][0]["message"]["content"],
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output


@register_api("anthropic")
def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
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
            output = {
                "answer": response.content[0].text
            }
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


@register_api("anthropic_thinking")
def chat_completion_anthropic_thinking(model, messages, max_tokens, budget_tokens, **kwargs):
    import anthropic

    client = anthropic.Anthropic(
        timeout=1200,
    )
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                },
                messages=messages,
            )
            output = {
                "thought": response.content[0].thinking,
                "answer": response.content[1].text,
            }
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


@register_api("mistral")
def chat_completion_mistral(model, messages, temperature, max_tokens, **kwargs):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = {
                "answer": chat_response.choices[0].message.content
            }
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


@register_api("xai")
def chat_completion_xai(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import xai_sdk

    client = xai_sdk.Client(api_key=api_dict['api_key'], api_host=api_dict['api_base']).compat
    output = API_ERROR_OUTPUT
    
    for _ in range(API_MAX_RETRY):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            )
            output = ""
            for chunk in stream:
                output += chunk.choices[0].delta.content
                
            break
        except Exception as e:
            print(type(e), e)
    

@register_api("gemini")
def http_completion_gemini(model, messages, **kwargs):
    import requests
    
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    sys_prompt = None
    if messages[0]["role"] == "system":
        sys_prompt = {
            "parts":[
                {"text": messages[0]["content"]}
                ]
            }
        messages = messages[1:]
        
    role_map = {"user": "user",
                "assistant": "model"}
    
    conv = [{"parts":[{"text":turn["content"]}], "role":role_map[turn["role"]]} for turn in messages]
    
    json_request = {
            "contents": conv,
            "safetySettings": safety_settings,
            "systemInstruction": sys_prompt,
    }

    if "temperature" in kwargs and "max_tokens" in kwargs:
        gen_config = {
            "temperature": kwargs["temperature"],
            "maxOutputTokens": kwargs["max_tokens"],
        }
        json_request["generationConfig"] = gen_config
    elif "temperature" in kwargs:
        gen_config = {
            "temperature": kwargs["temperature"],
        }
        json_request["generationConfig"] = gen_config
    elif "max_tokens" in kwargs:
        gen_config = {
            "maxOutputTokens": kwargs["max_tokens"],
        }
        json_request["generationConfig"] = gen_config
        
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json=json_request,
            )
        except Exception as e:
            print(f"**API REQUEST ERROR** Reason: {e}.")
            time.sleep(API_RETRY_SLEEP)
        if response.status_code != 200:
            print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")
            time.sleep(API_RETRY_SLEEP)
        try:
            output = {
                "answer": response.json()["candidates"][0]["content"]["parts"][0]["text"],
            }
        except KeyError as e:
            print(type(e), e)
            print(response.json())
    return output
    

@register_api("vertex")
def vertex_completion_gemini(model, messages, project_id, regions, **kwargs):
    import requests
    import subprocess
    
    output = API_ERROR_OUTPUT
    
    # Obtain the access token using gcloud CLI
    access_token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"], 
        text=True
    ).strip()

    if messages[0]["role"] == "system":
        data = {
            "systemInstruction": {
                "role": "system", # ignored by vertexi api (04/18/2025)
                "parts": [{
                    "text": messages[0]["content"]
                }]
            },
        }
        messages = messages[1:]
    else:
        data = {}
        
    role_map = {
        "user": "user",
        "assistant": "model"
    }
    
    messages = [{"parts":[{"text":turn["content"]}], "role":role_map[turn["role"]]} for turn in messages]

    url = (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{regions}/publishers/google/models/"
        f"{model}:generateContent"
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    data = data | {
        "contents": messages,
    } 
    
    if "temperature" in kwargs or "max_tokens" in kwargs:
        gen_config = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            gen_config["maxOutputTokens"] = kwargs["max_tokens"]
        data["generationConfig"] = gen_config

    response = requests.post(url, json=data, headers=headers)
    
    try:
        output = {
            "answer": response.json()["candidates"][0]["content"]["parts"][0]["text"],
        }
    except KeyError as e:
        print(type(e), e)
        print(response.json())
        
    return output


@register_api("cohere")
def chat_completion_cohere(model, messages, temperature, max_tokens, **kwargs):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = {
                "answer": response.text
            }
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


@register_api("meta")
def chat_completion_meta(model, messages, temperature, max_tokens, api_dict, **kwargs):
    assert api_dict
    texts = [{"role": m["role"],
              "text": m["content"]} for m in messages]
            
    output = ""
    for _ in range(API_MAX_RETRY):
        try:
            res = requests.post(
                f"{api_dict['api_base']}/chat_stream_completions?access_token={api_dict['api_key']}",
                stream=True,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "chunks_delimited": True,
                    "messages": texts,
                    "options": {
                        "max_tokens": max_tokens,
                        "generation_algorithm": "top_p",
                        "top_p": 1,
                        "temperature": temperature,
                    },
                },
                timeout=30,
            )
            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        part = json.loads(line.decode("utf-8"))
                        if "text" in part:
                            output += part["text"]
                break
            else:
                print(f"**API REQUEST ERROR** Code: {res.status_code}")
                time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print("**API REQUEST ERROR** Reason: Unknown.")
            time.sleep(API_RETRY_SLEEP)
            continue
        
    return {
        "answer": output
    }


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["uid"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def batch_submit_sglang(
    executor, 
    tokenizer, 
    temperature, 
    max_tokens, 
    all_context,
    max_context_length=None,
    end_think_token=None,
):
    print(f"DEBUG: sglang_completion_qwq: max_context_length: {max_context_length}")
    
    sampling_params = {
        "temperature": temperature,
        "skip_special_tokens": False,
        "max_new_tokens": max_tokens - 1,
        "no_stop_trim": True,
    }
        
    batch_prompt_token_ids = []
    batch_uids =[]
    uid_to_prompt = {}
    uid_to_response = {}
    
    for context in all_context:
        prompt_token_ids = tokenizer.apply_chat_template(
            context['turns'],
            add_generation_prompt=True,
            tokenize=True,
        )
        
        if max_context_length and (len(prompt_token_ids) + max_tokens) > max_context_length:
            print(f"DEBUG: sglang_completion_qwq: context length ({len(prompt_token_ids) + max_tokens}) > max_context_length ({max_context_length}), skip this context")
            continue
        
        batch_prompt_token_ids.append(prompt_token_ids)
        batch_uids.append(context['uid'])
        
        uid_to_prompt[context['uid']] = context['turns']
        
    err_msg = f"ERROR: len(batch_prompt_token_ids): {len(batch_prompt_token_ids)} != len(batch_uids): {len(batch_uids)}"
    assert len(batch_prompt_token_ids) == len(batch_uids), err_msg
    
    _ = executor.submit(
        prompt_token_ids=batch_prompt_token_ids,
        sampling_params=[sampling_params] * len(batch_uids),
        keys=batch_uids,
    )
    
    for request in tqdm(executor.as_completed(), total=len(batch_uids)):
        uid = request.key()
        result = request.result()
        raw_response = tokenizer.decode(
            result['output_ids'],
            skip_special_tokens=True,
        )
        
        if end_think_token:
            thought, _, ans = raw_response.partition(end_think_token)
            if ans == "":
                uid_to_response[uid] = {"thought": thought, "answer": raw_response}
            else:
                uid_to_response[uid] = {"thought": thought, "answer": ans}
        else:
            uid_to_response[uid] = {"answer": raw_response}
    
    # assert len(uid_to_response) == len(all_context), f"ERROR: len output ({len(uid_to_response)}) != len input ({len(all_context)})"
    return uid_to_response


def _infer_cuda_tp_world_size():
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is None:
        tp_world_size = 8
    else:
        tp_world_size = len(cuda_devices.split(","))
    return tp_world_size


def download_model(model: str, max_workers: int = 64):
    import subprocess
    
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    cmd = [
        "huggingface-cli", 
        "download", 
        f"--max-workers={max_workers}", 
        model
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Successfully downloaded model '{model}' with {max_workers} max workers.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading the model: {e}")


@register_engine("sglang")
def sglang_completion(
    model, 
    batch_context,
    answer_file,
    temperature,
    max_tokens=32768,
    end_think_token=None,
    **kwargs,
):
    from transformers import AutoTokenizer
    from utils.sglang_server import SGLangServerExecutor
    from utils.add_markdown_info import count_markdown_elements, remove_pattern
    import tiktoken
    import re

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    uids = [context['uid'] for context in batch_context]
    prompts = [context['prompt'] for context in batch_context]
    processed_context = [
        {
            "uid": uids[i], 
            "turns": [{
                "content": prompts[i],
                "role": "user",
            }]
        } 
        for i in tqdm(range(len(uids)))
    ]
    download_model(model=model)
    
    server_args = {
        "model_path": model,
        "dtype": "auto",
        "tp_size": _infer_cuda_tp_world_size(),
        "mem_fraction_static": 0.7,
        "max_prefill_tokens": max_tokens,
        "max_workers": 256,
        "server_port": 30000,
    }
    
    executor = SGLangServerExecutor(
        **server_args,
    )
    
    print(f"DEBUG: sglang_completion_qwq: model: {model}")
    
    uid_to_response = batch_submit_sglang(
        executor=executor, 
        tokenizer=tokenizer,
        temperature=temperature,
        max_tokens=max_tokens,
        all_context=processed_context,
        end_think_token=end_think_token,
    )
    
    executor.join()
    print("DEBUG: sglang_completion_qwq: done, sleep 10 seconds...")
    time.sleep(10)
        
    num_null = sum(
        [uid_to_response[uid]['answer'] is None for uid in uids if uid in uid_to_response]
    )
    print(f"Number of null responses: {num_null}")
    
    df = pd.DataFrame()
    df['uid'] = [context['uid'] for context in processed_context if context['uid'] in uid_to_response]
    df['ans_id'] = [shortuuid.uuid() for _ in range(len(df))]
    df['model'] = model
    df['messages'] = [
        context['turns'] + [
            {"content": uid_to_response[context['uid']], "role": "assistant"}
        ]
        for context in processed_context if context['uid'] in uid_to_response
    ]
    df['tstamp'] = [time.time() for _ in range(len(df))]
    
    encoding = tiktoken.encoding_for_model("gpt-4o")
    metadata = [
        {
            "token_len": len(encoding.encode(uid_to_response[context['uid']]['answer'], disallowed_special=()))
        } | count_markdown_elements(
            remove_pattern(
                uid_to_response[context['uid']]['answer'], 
                re.compile("```([^`]*)```")),
                suffix="",
        )
        for context in processed_context if context['uid'] in uid_to_response
    ]
    df["metadata"] = metadata 
    
    df.to_json(answer_file, lines=True, orient="records", force_ascii=False)
    
    pass
