## BenchBuilder

An automatic pipeline to create high-quality benchmark. BenchBuilder was used on Chatbot Arena data to curate Arena-Hard-v0.1

BenchBuilder employs a two stage pipeline.

Step 1: annotate the prompt using GPT-3.5-Turbo and filter prompts which either have a score < 5 or belong to a topic cluster with a mean score < 3. This serves as a cheap and first pass through to remove any low quality prompts and clusters before further curation. 

Step 2: use GPT-4-Turbo to annotate the remaining prompts, then extract prompts with quality score of >= 6 and belong to a topic cluster with mean quality score >= 6, ensuring only high-quality prompts are selected with minimal false positives.

After BenchBuilder, we stratified sampled multiple prompts per cluster to create a benchmark. However, you may employ whatever sampling scheme on prompts produced by BenchBuilder.

Checkout our [paper](https://arxiv.org/abs/2406.11939) for more details.