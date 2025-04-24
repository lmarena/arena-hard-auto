<div align="center">

# Arena-Hard-Auto

[![Github](https://img.shields.io/badge/Arena--Hard-black?logo=github&logoColor=white&labelColor=black&color=black)](https://github.com/lmarena/arena-hard-auto) [![arXiv](https://img.shields.io/badge/arXiv-Arena--Hard-b31b1b.svg)](https://arxiv.org/abs/2406.11939) [![Hugging Face Collection](https://img.shields.io/badge/Arena--Hard-fcd022?logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/collections/lmarena-ai/arena-hard-auto-680998796296d1462c729b6c) [![Twitter](https://img.shields.io/badge/LMArena--ai-white?logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/lmarena_ai)


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#leaderboard" style="text-decoration: none; font-weight: bold;">Leaderboard</a> •
    <a href="#install-dependencies" style="text-decoration: none; font-weight: bold;">Install</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">Evaluation</a> •
    <a href="https://huggingface.co/spaces/lmarena-ai/arena-hard-viewer" style="text-decoration: none; font-weight: bold;">Demo</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a>
  </p>
</div>

</div>

# News
- **[Apr 23, 2025]** 🎉 **Arena-Hard-v2.0** is finally here! Better judges, new hard prompts, and additional eval for creative writing.
- **[Oct 14, 2024]** 🎉 **Style Control** is now supported in Arena-Hard-Auto.

## About

Arena-Hard-Auto is an automatic evaluation tool for instruction-tuned LLMs. Arena-Hard-Auto has the highest correlation and separability to LMArena (Chatbot Arena) among popular open-ended LLM benchmarks ([See Paper](https://arxiv.org/abs/2406.11939)). If you are curious to see how well your model might perform on LMArena before deploying, we recommend trying Arena-Hard-Auto's newest evaluation set, **Arena-Hard-v2.0-Preview**.

V2.0 contains 500 fresh, challenging real-world user queries (open-ended software engineering problems, math questions, etc) and 250 creative writing queries sourced from Chatbot Arena. We employs automatic judges, GPT-4.1 and Gemini-2.5, as a cheaper and faster approximator to human preference.

Although both Arena-Hard-Auto and Chatbot Arena Category Hard ([See Blog](https://lmsys.org/blog/2024-05-17-category-hard/)) employ similar pipeline to select hard prompts, Arena-Hard-Auto employs automatic judge as a cheaper and faster approximator to human preference. Checkout [BenchBuilder](BenchBuilder) folder for code and resources on how we curate Arena-Hard-Auto. In the paper we also purposed metrics, such as model separability and agreement to human preference, for evaluating benchmarks' ability to rank models (See [Evaluate Benchmarks](#evaluate-benchmarks) for more information and code).


## Leaderboard

### Arena-Hard-v2.0-Preview

Hard Prompt, Style Control, and Gemini-2.5 as Judge **(Official Configuration)**:
```console
                                      Model  Scores (%)         CI (%)
0                             o3-2025-04-16        86.1  (-1.1 / +1.1)
1                                gemini-2.5        79.3  (-1.5 / +1.9)
2                   o4-mini-2025-04-16-high        79.2  (-1.2 / +1.5)
3                        o4-mini-2025-04-16        74.8  (-1.4 / +1.4)
4                          gemini-2.5-flash        69.0  (-1.3 / +1.9)
5                   o3-mini-2025-01-31-high        66.5  (-1.9 / +1.4)
6   claude-3-7-sonnet-20250219-thinking-16k        61.1  (-2.1 / +1.5)
7                        o1-2024-12-17-high        61.0  (-1.6 / +1.8)
8                               deepseek-r1        57.9  (-2.4 / +2.3)
9                             o1-2024-12-17        56.0  (-1.7 / +2.0)
10                          gpt-4.5-preview        50.7  (-1.8 / +1.7)
11                                  gpt-4.1        50.7  (-2.3 / +1.9)
12                       o3-mini-2025-01-31        50.0  (-0.0 / +0.0)
13                             gpt-4.1-mini        47.2  (-1.9 / +2.6)
14                                  QwQ-32B        43.7  (-2.4 / +2.1)
15               claude-3-5-sonnet-20241022        33.6  (-1.9 / +1.7) 
16                                 s1.1-32B        22.2  (-1.6 / +1.6) 
17           llama4-maverick-instruct-basic        17.5  (-1.4 / +1.6) 
18                           Athene-V2-Chat        16.5  (-1.0 / +1.5) 
19                           gemma-3-27b-it        14.8  (-1.3 / +0.9) 
20                             gpt-4.1-nano        14.1  (-1.3 / +1.0) 
21       Llama-3.1-Nemotron-70B-Instruct-HF        10.1  (-0.9 / +0.8) 
22                     Qwen2.5-72B-Instruct        10.1  (-0.8 / +1.3) 
23                         OpenThinker2-32B         3.1  (-0.2 / +0.4)
```

Hard Prompt, Style Control, and GPT-4.1 as Judge **(If prefer OpenAI API)**
```console
                                      Model  Scores (%)         CI (%)
0                             o3-2025-04-16        87.1  (-1.1 / +1.0)
1                   o4-mini-2025-04-16-high        81.9  (-1.7 / +1.2)
2                        o4-mini-2025-04-16        78.1  (-1.6 / +1.5)
3                   o3-mini-2025-01-31-high        65.4  (-2.2 / +2.3)
4                                   gpt-4.1        59.0  (-2.4 / +2.2)
5                        o1-2024-12-17-high        58.4  (-2.2 / +1.9)
6                        o3-mini-2025-01-31        50.0  (-0.0 / +0.0)
7                              gpt-4.1-mini        49.4  (-2.6 / +2.2)
8                             o1-2024-12-17        49.4  (-1.8 / +2.6)
9                                gemini-2.5        48.9  (-2.1 / +2.2)
10                              deepseek-r1        48.3  (-2.1 / +2.5)
11  claude-3-7-sonnet-20250219-thinking-16k        48.2  (-2.8 / +2.1)
12                         gemini-2.5-flash        44.5  (-2.0 / +2.5)
13                          gpt-4.5-preview        43.8  (-2.2 / +2.0)
14                                  QwQ-32B        36.7  (-2.0 / +2.0)
15               claude-3-5-sonnet-20241022        26.3  (-2.2 / +2.3)
16                                 s1.1-32B        18.5  (-2.3 / +2.2)
17                             gpt-4.1-nano        15.5  (-1.4 / +1.4)
18                           Athene-V2-Chat        12.8  (-1.1 / +1.3)
19           llama4-maverick-instruct-basic        12.4  (-1.0 / +1.3)
20                           gemma-3-27b-it         9.5  (-0.7 / +0.9)
21                     Qwen2.5-72B-Instruct         7.9  (-0.9 / +1.0)
22       Llama-3.1-Nemotron-70B-Instruct-HF         6.8  (-0.9 / +0.6)
23                         OpenThinker2-32B         2.2  (-0.2 / +0.3)
```

Creative Writing, Ensemble GPT-4.1 and Gemini 2.5 as Judges **(Best Configuration for Creative Writing)**
```console
                                      Model  Scores (%)         CI (%)
0                                gemini-2.5        90.8  (-1.0 / +1.2)
1                             o3-2025-04-16        88.8  (-1.2 / +1.1)
2                          gemini-2.5-flash        83.8  (-1.6 / +1.2)
3                               deepseek-r1        77.2  (-1.6 / +1.7)
4                            gemma-3-27b-it        69.8  (-1.5 / +1.8)
5   claude-3-7-sonnet-20250219-thinking-16k        63.9  (-1.9 / +1.8)
6                                   gpt-4.1        61.6  (-1.7 / +1.7)
7                                   QwQ-32B        61.2  (-1.9 / +1.8)
8                        o1-2024-12-17-high        60.0  (-1.7 / +1.5)
9                   o4-mini-2025-04-16-high        58.5  (-1.5 / +1.8)
10                            o1-2024-12-17        56.4  (-1.5 / +1.8)
11                       o4-mini-2025-04-16        55.5  (-1.5 / +1.6)
12                          gpt-4.5-preview        51.4  (-2.4 / +1.9)
13                     gemini-2.0-flash-001        50.0  (-0.0 / +0.0)
14                  o3-mini-2025-01-31-high        42.9  (-2.0 / +2.0)
15                             gpt-4.1-mini        28.0  (-2.1 / +1.8)
16       Llama-3.1-Nemotron-70B-Instruct-HF        27.3  (-1.6 / +1.5)
17               claude-3-5-sonnet-20241022        24.1  (-1.8 / +1.6)
18                         OpenThinker2-32B        23.6  (-1.1 / +1.4)
19                           Athene-V2-Chat        17.9  (-1.2 / +1.3)
20                             gpt-4.1-nano        10.7  (-1.2 / +1.3)
21           llama4-maverick-instruct-basic        10.4  (-1.0 / +1.0)
22                     Qwen2.5-72B-Instruct        10.2  (-1.3 / +1.1)
23                                 s1.1-32B         8.2  (-0.8 / +0.8)
```

For older leaderboards, such as Arena-Hard-v0.1, see [past-leaderboards](/misc/past_leaderboards.md)

## Install Dependencies
```
git clone https://github.com/lmarena/arena-hard-auto.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

## Download dataset
We have pre-generated many popular models answers and judgments. You can browse them with an online [demo](https://huggingface.co/spaces/lmarena-ai/arena-hard-viewer) or download them (with [`git-lfs`](https://git-lfs.com) installed) by
```console
> git lfs install
> git clone git@hf.co:datasets/lmarena-ai/arena-hard-auto arena-hard-data
// copy answers/judgments to the data directory
> cp -r arena-hard-data/data . 
```

Then run
```console
> python show_result.py
                                      Model  Scores (%)         CI (%)
0                             o3-2025-04-16        87.6  (-0.8 / +1.0)
1                   o4-mini-2025-04-16-high        82.7  (-1.4 / +1.3)
2                        o4-mini-2025-04-16        78.9  (-1.6 / +1.6)
```

## Evaluate

### Step 1. Set up the endpoint config to your model

Fill in your API endpoint in `config/api_config.yaml`. We support OpenAI compatible API server, Anthropic, Vertex AI, and more. You will find examples in `config/api_config.yaml`.

You may use inference engine such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [SGLang](https://github.com/sgl-project/sglang?tab=readme-ov-file#using-local-models) to host your model with an OpenAI compatible API server.

We also include support for fast built-in inference with SGLang, see examples in `config/api_config.yaml` and implementaton in `utils/completion.py`. See `misc/sglang_setup.bash` for environment setup.

### Step 2. Generate Model Answers

In `config/gen_answer_config.yaml`, add your model name in `model_list`.

Run the command to generate answers:
```console
> python gen_answer.py
```

Caching feature is implemented. The code will skip generating an answer when there is already an existing answer/judgment to the same prompt (this feature is not supported for built-in SGLang server).

### Step 3. Generate Judgments

In `config/arena-hard-v2.0.yaml`, add your model name in `model_list`.
```yaml
...
# Add your model below for evaluation
model_list:
  - deepseek-r1
  - [YOUR-MODEL-NAME]
```

We recommend employing GPT-4.1 as judge for fast, stable judge inference. To use Gemini-2.5, comment out:
```yaml
judge_model: gpt-4.1
temperature: 0.0
max_tokens: 16000
```

and uncomment:
```yaml
judge_model: gemini-2.5
temperature: 1.0
max_tokens: 32000
```

Run the command to generate judgments:
```console
> python gen_judgment.py
```

For Ensemble-as-Judges, we suggest inferencing both judges independently and we will aggregrate the results when displaying leaderboard for you (see step 4).

Judgment caching is also implemented. It will skip generating judgments that has already been generated or lacks one of the model answers.  

### Step 4. Show result
Output model win rates for **Arena-Hard-v2.0-Preview (Hard Prompt, Style Control, GPT-4.1 as Judge)**:
```console
> python show_result.py --judge-names gpt-4.1 --control-features markdown length
```

Output model win rates for **Arena-Hard-v2.0-Preview (Creative Writing, Ensemble GPT-4.1 and Gemini 2.5 as Judges)**:
```console
> python show_result.py --judge-names gpt-4.1 gemini-2.5 --category creative_writing
```

### Step 5. Benchmark Viewer
You can review answers and judgment results using our gradio script (`gradio>=5.25.2`).
```console
> python qa_browser.py --share
```

## Style Control
Following the newly introduced Style Control on Chatbot Arena, we release Style Control on Arena Hard Auto! We employ the same Style Control methods as proposed in the [blogpost](https://lmsys.org/blog/2024-08-28-style-control/). Please refer to the blogpost for methodology and technical background.

Before applying style control, make sure your model answers has proper style attribute generated. Either pull the latest data from [huggingface repo](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto), or run the following script!

To add style attribute to your model answers, use `add_markdown_info.py`. The following command takes model answers from `--dir`, append style attributes (token length, number of headers, etc), and save the new answers in `--output-dir`.

```console
> python add_markdown_info.py --dir data/arena-hard-v0.1/model_answer --output-dir data/arena-hard-v0.1/model_answer
```

To control for style (token length and markdown elements), use `--control-features` or `-f` when running `show_result.py`.

```console
> python show_result.py -f markdown length # style control
> python show_result.py -f markdown # control for markdown density only
> python show_result.py -f length # length control only
```

## Evaluate Benchmarks
We outline two key properties that the benchmark aiming to approximate human preference should possess to provide meaningful comparisons between models:
1. Separability: the benchmark should separate models with high confidence.
2. Alignment with Human Preference: the benchmark should agree with human preference.

While previous works have focused on alignment, separability is also a crucial consideration when comparing models of similar quality (e.g., different checkpoints from the same training run). However, achieving high-confidence separability is challenging due to limitations in prompt design and inherent variances in LLM evaluations. Overly simplistic prompts fail to distinguish between models, while the randomness in human and LLM judgments leads to inconsistent predictions. As a result, it is often difficult to confidently determine if a model’s apparent performance reflects a genuine difference in capability or merely noisy observations, highlighting a need for methods to verify whether a benchmark can reliably separate similar models.

Statistical measures like Pearson (Pearson, 1895) and Spearman Correlations (Spearman, 1961), commonly used in benchmarks such as AlpacaEval (Li et al., 2023) to measure correlation to human preference ranking, may fail to adequately address model separability and ranking instability. In addition, these measures only provide a coarse signal of ranking correlation without quantifying the magnitude of performance differences between model pairs. To address these shortcomings, we develop three novel metrics: **Separability with Confidence**, **Agreement with Confidence**, and **Pair Rank Brier Score**.

**Separability with Confidence** quantifies the benchmark’s confidence by measuring its consistency in predicting the winner of a model pair across random seeds through bootstrapping. This is done by calculating the percentage of model pairs that have non-overlapping confidence intervals of their benchmark scores. A higher percentage indicates that the benchmark is more confident in distinguishing between the performance of different models, as the confidence intervals of their scores do not overlap.

For **Agreement with Confidence**, and **Pair Rank Brier Score**, please refer to section 3 of our [paper](https://arxiv.org/abs/2406.11939). The code for calculating these metrics can be found in this [colab notebook](https://colab.research.google.com/drive/1ar6XLWREN_dXEh404WNOxroFVUe_4njp). 

## Community Contribution

Feel free to submit a PR or open up an issue!

If you want to add your model to the leaderboard, please email me the following:
1. An OpenAI compatible endpoint to your model.
2. An OpenAI API key for me to inference judgment.

Sorry for the inconvience! Since Arena-Hard-Auto is open data, we want to avoid people cheating on our leaderboard. If we find anything suspicious, we reserve the right to not add your model to our leaderboard.

## Citation
The code in this repository is developed from the papers below. Please cite it if you find the repository helpful.
```
@article{li2024crowdsourced,
  title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline},
  author={Li, Tianle and Chiang, Wei-Lin and Frick, Evan and Dunlap, Lisa and Wu, Tianhao and Zhu, Banghua and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.11939},
  year={2024}
}
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```
