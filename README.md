# Arena-Bench-v1
World's Hardest LLM Benchmark! 

### Evaluate models using arena-bench-v1:
To generate model answers using FastChat.
```console
cd benchmark
python -m fastchat.llm_judge.gen_api_answer --bench-name arena-bench-v1 --model gpt-3.5-turbo
```
To generate reference answers for GPT-4 judge.
```console
python -m fastchat.llm_judge.gen_api_answer --bench-name arena-bench-v1 \
          --model gpt-4 \
          --answer-file data/arena-bench-v1/reference_answer/gpt-4.jsonl
```
To generate GPT-4 judgment.
```console
python ../judge_any.py --config-file data/arena-bench-v1_config.yaml \
          --bench-name arena-bench-v1 \
          --mode all
```
Evaluate arena-bench-v1 scores.
```console
python benchmark_eval.py --bench-name arena-bench-v1
```

### Creating Challenging Benchmark Questions
To evaluate how challenging benchmark prompts are.
```console
cd benchmark
python ../judge_any.py --config-file data/prompt-judge_config.yaml \
          --bench-name prompt-judge \
          --mode question
```
