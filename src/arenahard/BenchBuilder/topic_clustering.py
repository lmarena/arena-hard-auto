
import json
import argparse
import os

import spacy.cli
import torch
from tqdm import tqdm

import numpy as np

import openai
import tiktoken

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from bertopic.backend import OpenAIBackend


def run(args):
    client = openai.OpenAI()

    if args.embedding_file is not None:
        embeddings = np.load(args.embedding_file)
        if args.post_process_conv is not None:
            all_convs = json.load(open(args.post_process_conv))
        else:
            raise ValueError("Please provide post process conv file")

        convs = []
        for row in all_convs:
            convs.append(row["post_process_conv"])
    else:
        all_convs = json.load(open(args.conv_file))

        if args.first_n is not None:
            all_convs = all_convs[:args.first_n]

        all_convs_new = []
        convs = []
        for row in all_convs:
            if row["language"] != "English":
                continue
            conv = ""
            for turns in row["conversation_a"]:
                if turns["role"] == "user":
                    content = turns["content"]
                    if isinstance(content, list):
                        conv += f"{content[0]}\n"
                    elif isinstance(content, str):
                        conv += f"{content}\n"
                    else:
                        raise ValueError(f"Unknown content type: {type(content)}")
            
            conv = conv.replace("<|endoftext|>", "<| endoftext |>")
            if len(conv) <= 32:
                continue
            convs.append(conv[:10000])
            row["post_process_conv"] = conv[:10000]
            all_convs_new.append(row)

        # save convs to file
        with open(f"{args.output_dir}/post_process_convs.json", "w") as f:
            json.dump(all_convs_new, f, indent=4)

        batch_size = 2000
        embeddings = []
        for i in tqdm(range(0, len(convs), batch_size)):
            convs_sub = convs[i : i + batch_size]
            responses = client.embeddings.create(input=convs_sub, model="text-embedding-3-small").data
            embeddings.extend([data.embedding for data in responses])
        embeddings = torch.tensor(embeddings)
        embeddings = embeddings.numpy()

        # save embedding to numpy file
        np.save(f"{args.output_dir}/embeddings.npy", embeddings)

    print("#convos:", len(convs))
        
    # Part-of-Speech
    try:
        pos_model = PartOfSpeech("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        pos_model = PartOfSpeech("en_core_web_sm")
    # GPT
    tokenizer= tiktoken.encoding_for_model("gpt-4o-mini")

    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    topic: <topic label>
    """
    openai_model = OpenAI(client, model="gpt-4o-mini", exponential_backoff=True, chat=True, prompt=prompt, nr_docs=50, doc_length=500, tokenizer=tokenizer)

    # All representation models
    representation_model = {
        "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "POS": pos_model
    }

    embedding_model = OpenAIBackend(client, "text-embedding-3-small", delay_in_seconds=3, batch_size=100)

    topic_model = BERTopic(
        verbose=True,
        embedding_model=embedding_model,
        representation_model=representation_model,
        min_topic_size=args.min_topic_size,
    )
    topics, _ = topic_model.fit_transform(convs, embeddings); len(topic_model.get_topic_info())

    new_topics = topic_model.reduce_outliers(convs, topics)
    with open(f"{args.output_dir}/conv_topics.json", "w") as f:
        json.dump(new_topics, f, default=str)

    topic_model.save(f"{args.output_dir}/model_dir", serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)
    
    df = topic_model.get_topic_info()
    df.to_csv(f"{args.output_dir}/topics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-file", type=str, required=True)
    parser.add_argument("--min-topic-size", type=int, default=32)
    parser.add_argument("--embedding-file", type=str, default=None)
    parser.add_argument("--post-process-conv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="topic_model_dir")
    parser.add_argument("--first-n", type=int, default=None)
    args = parser.parse_args()
    
    # create dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    run(args)
