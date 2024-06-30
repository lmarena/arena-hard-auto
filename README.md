# Arena-Hard-Auto
Arena-Hard-Auto-v0.1 is an automatic evaluation tool for instruction-tuned LLMs. It contains 500 challenging user queries. We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314). Although both Arena-Hard-Auto and Chatbot Arena Category Hard employ similar pipeline to select hard prompts, Arena-Hard-Auto employs automatic judge as a cheaper and faster approximator to human preference. Notably, Arena-Hard-Auto has the highest correlation and separability to Chatbot Arena among popular open-ended LLM benchmarks (see our paper). If you are curious to see how well your model might perform on Chatbot Arena, we recommend trying Arena-Hard-Auto. 

Check out our paper for more details about how Arena Hard Auto v0.1 works -> [Paper link](https://arxiv.org/abs/2406.11939).

## Full Leaderboard (Updated: 06/29)
```console
claude-3-5-sonnet-20240620     | score: 79.3  | 95% CI: (-2.1, 2.0)  | average #tokens: 567
gpt-4o                         | score: 79.2  | 95% CI: (-1.9, 1.7)  | average #tokens: 696          
gpt-4-0125-preview             | score: 78.0  | 95% CI: (-2.1, 2.4)  | average #tokens: 619
gemini-1.5-pro-api-preview     | score: 72.0  | 95% CI: (-2.1, 2.5)  | average #tokens: 676
glm-4-0520                     | score: 63.8  | 95% CI: (-2.9, 2.8)  | average #tokens: 636          
yi-large                       | score: 63.7  | 95% CI: (-2.6, 2.4)  | average #tokens: 626
deepseek-coder-v2              | score: 62.3  | 95% CI: (-2.1, 1.8)  | average #tokens: 578             
claude-3-opus-20240229         | score: 60.4  | 95% CI: (-2.5, 2.5)  | average #tokens: 541
gemma-2-27b-it                 | score: 57.5  | 95% CI: (-2.1, 2.4)  | average #tokens: 577 
glm-4-0116                     | score: 55.7  | 95% CI: (-2.4, 2.3)  | average #tokens: 622
glm-4-air                      | score: 50.9  | 95% CI: (-2.9, 2.7)  | average #tokens: 619
gpt-4-0314                     | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 423
gemini-1.5-flash-api-preview   | score: 49.6  | 95% CI: (-2.2, 2.8)  | average #tokens: 642
qwen2-72b-instruct             | score: 46.9  | 95% CI: (-2.5, 2.7)  | average #tokens: 515          
claude-3-sonnet-20240229       | score: 46.8  | 95% CI: (-2.3, 2.7)  | average #tokens: 552
claude-3-haiku-20240307        | score: 41.5  | 95% CI: (-2.5, 2.5)  | average #tokens: 505
llama-3-70b-chat-hf            | score: 41.1  | 95% CI: (-2.0, 2.2)  | average #tokens: 583
gpt-4-0613                     | score: 37.9  | 95% CI: (-2.8, 2.4)  | average #tokens: 354
mistral-large-2402             | score: 37.7  | 95% CI: (-2.1, 2.6)  | average #tokens: 400
mixtral-8x22b-instruct-v0.1    | score: 36.4  | 95% CI: (-2.4, 2.6)  | average #tokens: 430
Qwen1.5-72B-Chat               | score: 36.1  | 95% CI: (-2.0, 2.7)  | average #tokens: 474
phi-3-medium-4k-instruct       | score: 33.4  | 95% CI: (-2.6, 2.1)  | average #tokens: 517          
command-r-plus                 | score: 33.1  | 95% CI: (-2.8, 2.4)  | average #tokens: 541
mistral-medium                 | score: 31.9  | 95% CI: (-1.9, 2.2)  | average #tokens: 485
phi-3-small-8k-instruct        | score: 29.8  | 95% CI: (-1.8, 1.9)  | average #tokens: 568          
mistral-next                   | score: 27.4  | 95% CI: (-2.4, 2.4)  | average #tokens: 297
gpt-3.5-turbo-0613             | score: 24.8  | 95% CI: (-1.9, 2.3)  | average #tokens: 401
claude-2.0                     | score: 24.0  | 95% CI: (-1.8, 1.8)  | average #tokens: 295
dbrx-instruct                  | score: 23.9  | 95% CI: (-1.5, 1.5)  | average #tokens: 415
Mixtral-8x7B-Instruct-v0.1     | score: 23.4  | 95% CI: (-2.0, 1.9)  | average #tokens: 457
gpt-3.5-turbo-0125             | score: 23.3  | 95% CI: (-2.2, 1.9)  | average #tokens: 329
Yi-34B-Chat                    | score: 23.1  | 95% CI: (-1.6, 1.8)  | average #tokens: 611
Starling-LM-7B-beta            | score: 23.0  | 95% CI: (-1.8, 1.8)  | average #tokens: 530
claude-2.1                     | score: 22.8  | 95% CI: (-2.3, 1.8)  | average #tokens: 290
Snorkel-Mistral-PairRM-DPO     | score: 20.7  | 95% CI: (-1.8, 2.2)  | average #tokens: 564                       
llama-3-8b-chat-hf             | score: 20.6  | 95% CI: (-2.0, 1.9)  | average #tokens: 585                       
gpt-3.5-turbo-1106             | score: 18.9  | 95% CI: (-1.8, 1.6)  | average #tokens: 285                       
gpt-3.5-turbo-0301             | score: 18.1  | 95% CI: (-1.9, 2.1)  | average #tokens: 334                               
gemini-1.0-pro                 | score: 17.8  | 95% CI: (-1.2, 2.2)  | average #tokens: 322                               
snowflake-arctic-instruct      | score: 17.6  | 95% CI: (-1.8, 1.5)  | average #tokens: 365                                         
command-r                      | score: 17.0  | 95% CI: (-1.7, 1.8)  | average #tokens: 432                                         
phi-3-mini-128k-instruct       | score: 15.4  | 95% CI: (-1.4, 1.4)  | average #tokens: 609                                                    
tulu-2-dpo-70b                 | score: 15.0  | 95% CI: (-1.6, 1.3)  | average #tokens: 550                                                    
Starling-LM-7B-alpha           | score: 12.8  | 95% CI: (-1.6, 1.4)  | average #tokens: 483                                                    
mistral-7b-instruct            | score: 12.6  | 95% CI: (-1.7, 1.4)  | average #tokens: 541                                                                 
gemma-1.1-7b-it                | score: 12.1  | 95% CI: (-1.3, 1.3)  | average #tokens: 341                                                                 
Llama-2-70b-chat-hf            | score: 11.6  | 95% CI: (-1.5, 1.2)  | average #tokens: 595                                                                 
vicuna-33b-v1.3                | score:  8.6  | 95% CI: (-1.1, 1.1)  | average #tokens: 451                                                                 
gemma-7b-it                    | score:  7.5  | 95% CI: (-1.2, 1.3)  | average #tokens: 378                                                                                
Llama-2-7b-chat-hf             | score:  4.6  | 95% CI: (-0.8, 0.8)  | average #tokens: 561                                                                                
gemma-1.1-2b-it                | score:  3.4  | 95% CI: (-0.6, 0.8)  | average #tokens: 316                                                                                
gemma-2b-it                    | score:  3.0  | 95% CI: (-0.6, 0.6)  | average #tokens: 369
```

## Install Dependencies
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

## Download dataset
We have pre-generated many popular models answers and judgments. You can browse them with an online [demo](https://huggingface.co/spaces/lmsys/arena-hard-browser) or download them (with [`git-lfs`](https://git-lfs.com) installed) by
```console
> git clone https://huggingface.co/spaces/lmsys/arena-hard-browser
// copy answers/judgments to the data directory
> cp -r arena-hard-browser/data . 
```
Then run
```console
> python show_result.py
gpt-4-0125-preview             | score: 78.0  | 95% CI: (-1.8, 2.2)  | average #tokens: 619
claude-3-opus-20240229         | score: 60.4  | 95% CI: (-2.6, 2.1)  | average #tokens: 541
gpt-4-0314                     | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 423
claude-3-sonnet-20240229       | score: 46.8  | 95% CI: (-2.7, 2.3)  | average #tokens: 552
claude-3-haiku-20240307        | score: 41.5  | 95% CI: (-2.4, 2.5)  | average #tokens: 505
gpt-4-0613                     | score: 37.9  | 95% CI: (-2.1, 2.2)  | average #tokens: 354
mistral-large-2402             | score: 37.7  | 95% CI: (-2.9, 2.8)  | average #tokens: 400
Qwen1.5-72B-Chat               | score: 36.1  | 95% CI: (-2.1, 2.4)  | average #tokens: 474
command-r-plus                 | score: 33.1  | 95% CI: (-2.0, 1.9)  | average #tokens: 541
```
Running `show_results.py` will save generated battles into `data/arena_hard_battles.jsonl` and bootstrapping statistics into `data/bootstrapping_results.jsonl`. If you don't want to regenerate battles or bootstrapping statistics, simply toggle argument `--load-battles` or `--load-bootstrap`, respectively.

## Evaluate a new model on Arena-Hard-Auto v0.1:

### Step 1. Set up the endpoint config to your model

Fill in your API endpoint in `config/api_config.yaml`. We support OpenAI compatible API server. You can specify `parallel` to indicate the number of concurrent API requests (default: 1).
```yaml
# example
gpt-3.5-turbo-0125:
    model_name: gpt-3.5-turbo-0125
    endpoints: null
    api_type: openai
    parallel: 8

[YOUR-MODEL-NAME]:
    model_name: [YOUR-MODEL-NAME]
    endpoints:
        - api_base: [YOUR-ENDPOINT-URL]
          api_key: [YOUR-API-KEY]
    api_type: openai
    parallel: 8
```
You may use inference engine such as [Latest TGI version](https://huggingface.co/docs/text-generation-inference/en/messages_api) or [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [SGLang](https://github.com/sgl-project/sglang?tab=readme-ov-file#using-local-models) to host your model with an OpenAI compatible API server.

TGI Quick start
```
hf_pat=
model=
volume=/path/to/cache
port=1996

huggingface-cli download $model
sudo docker run --gpus 8 -e HUGGING_FACE_HUB_TOKEN=$hf_pat --shm-size 2000g -p $port:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0.4 --model-id $model --max-input-length 8192 --max-batch-total-tokens 8193 --max-batch-prefill-tokens 8193 --max-total-tokens 8193
```

### Step 2. Generate Model Answers

In `config/gen_answer_config.yaml`, add your model name in `model_list`.
```yaml
bench_name: arena-hard-v0.1
temperature: 0.0
max_tokens: 4096
num_choices: 1


model_list:
  - [YOUR-MODEL-NAME]
```
Run the command to generate answers:
```console
python gen_answer.py
```
Caching feature is implemented. The code will skip generating an answer when there is already an existing answer/judgment to the same prompt. 

### Step 3. Generate Judgments

In `config/judge_config.yaml`, add your model name in `model_list`.
```yaml
...
# Add your model below for evaluation
model_list:
  - gpt-3.5-turbo-0125
  - [YOUR-MODEL-NAME]
```

Run the command to generate judgments:
```console
python gen_judgment.py
```
Judgment caching is also implemented. It will skip generating judgments that has already been generated or lacks one of the model answers.  

### Step 4. Show result
Output model win rates.  Optionally, use `--full-stats` for detailed results.
```console
> python show_result.py
```
### Step 5. Arena Hard UI
You can review individual judgment results using our UI code.
```console
> python qa_broswer.py --share
```

## Community Contribution
Coming soon...

## Citation
The code in this repository is mostly developed for or derived from the papers below. Please cite it if you find the repository helpful.
```
@misc{li2024crowdsourced,
      title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline}, 
      author={Tianle Li and Wei-Lin Chiang and Evan Frick and Lisa Dunlap and Tianhao Wu and Banghua Zhu and Joseph E. Gonzalez and Ion Stoica},
      year={2024},
      eprint={2406.11939},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@misc{chiang2024chatbot,
    title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
    author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
    year={2024},
    eprint={2403.04132},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```
