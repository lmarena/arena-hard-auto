# Arena-Hard
**Currently still under revisions. Blog and technical report will be released soon.**
Arena-Hard is an evaluation tool for instruction-tuned LLMs. It contains 500 challenging user queries. We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314). 

Check out our blog post for more details. 

## Install Dependencies
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

## Download dataset
We have pre-generated many popular models answers and judgments. You can browse them with an online [demo](https://huggingface.co/spaces/lmsys/arena-hard-browser) or download them by
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
Running show results will save generated battles into `data/arena_hard_battles.jsonl` and bootstrapping statistics into `data/bootstrapping_results.jsonl`. If you don't want to regenerate battles or bootstrapping statistics, simply toggle argument `--load-battles` or `--load-bootstrap`, respectively.

## Evaluate a new model on Arena-hard-v0.1:

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
You may use inference engine such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [SGLang](https://github.com/sgl-project/sglang?tab=readme-ov-file#using-local-models) to host your model with an OpenAI compatible API server.


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

## Citation
