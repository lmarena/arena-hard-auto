# Arena-Hard

Arena-Hard is an evaluation tool for instruction-tuned LLMs. It contains 500 challenging user queries. We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314).

Check out our blog post for more details.

## Install Dependencies
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

## Review pre-generated results:
We pre-generate many popular models answer and judgment. You can review them by:
```console
> python show_result.py

gpt-4-0125-preview             | win-rate: 77.74 | average #tokens: 618
claude-3-opus-20240229         | win-rate: 60.36 | average #tokens: 539
claude-3-sonnet-20240229       | win-rate: 47.24 | average #tokens: 553
claude-3-haiku-20240307        | win-rate: 41.47 | average #tokens: 504
gpt-4-0613                     | win-rate: 37.9  | average #tokens: 354
mistral-large-2402             | win-rate: 37.77 | average #tokens: 399
Qwen1.5-72B-Chat               | win-rate: 36.08 | average #tokens: 473
mistral-medium                 | win-rate: 32.94 | average #tokens: 492
gpt-3.5-turbo-0613             | win-rate: 25.14 | average #tokens: 403
```

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
