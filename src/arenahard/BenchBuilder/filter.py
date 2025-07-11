"""
Filter prompts based on scores and cluster thresholds. To be run after topic_clustering.py and label.py
"""
import hashlib
import os

import orjson
import json
import argparse
from typing import List, Dict
import numpy as np
import wandb

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'rb') as f:
        return orjson.loads(f.read())

def load_jsonl(file_path: str) -> List[Dict]:
    conversations = []
    with open(file_path, 'rb') as f:
        for line in f:
            conversations.append(orjson.loads(line))
    return conversations

def calculate_score(conversation: Dict) -> int:
    criteria = conversation.get('category_tag', {}).get('criteria_v0.1', {})
    return sum(1 for value in criteria.values() if value)

def calculate_cluster_scores(conversations: List[Dict], clusters: List[int]) -> Dict[int, float]:
    cluster_scores = {}
    for conv, cluster in zip(conversations, clusters):
        score = calculate_score(conv)
        if cluster not in cluster_scores:
            cluster_scores[cluster] = []
        cluster_scores[cluster].append(score)
    
    cluster_to_mean_score = {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}
    print(f"Cluster to mean score: {cluster_to_mean_score}")
    return cluster_to_mean_score

def filter_prompts(conversations: List[Dict], clusters: List[int], prompt_threshold: int, cluster_threshold: float) -> List[Dict]:
    cluster_scores = calculate_cluster_scores(conversations, clusters)
    
    filtered_prompts = []
    for conv, cluster in zip(conversations, clusters):
        score = calculate_score(conv)
        if score >= prompt_threshold and cluster_scores[cluster] >= cluster_threshold:
            conv.update({
                "prompt_score": score,
            })
            filtered_prompts.append(conv)
    
    return filtered_prompts

def to_arena_hard_questions_format(conversations: List[Dict], clusters: List[int], topics_file: str, image_dir: str) -> List[Dict]:
    """
    Convert to a format like this:
    {"question_id":"328c149ed45a41c0b9d6f14659e63599",
     "category":"arena-hard-v0.1",
     "cluster":"ABC Sequence Puzzles & Groups",
     "turns":[{"content":"Use ABC notation to write a melody in the style of a folk tune."}]
    }
    """

    topics_map = load_json(topics_file)
    cluster_number_to_name: Dict[str, str] = {}
    for cluster_number, cluster_obj in topics_map["topic_aspects"]["OpenAI"].items():
        cluster_number_to_name[cluster_number] = cluster_obj[0][0]

    arena_hard_questions = []
    for i, (conv, cluster) in enumerate(zip(conversations, clusters)):
        # Contains image
        if isinstance(conv["conversation_a"][0]["content"], list):
            image_hash = conv["conversation_a"][0]["content"][1][0]
            image_path = os.path.join(image_dir, f"{image_hash}.png")
            is_image_valid = os.path.exists(image_path)
            if not is_image_valid:
                print(f"Image not found: {image_path}, not included in benchmark.")
                continue
        
        turns_list = []
        turns_list.append({"content": conv["conversation_a"][0]["content"]})

        arena_hard_questions.append({
            "question_id": f"{i}",
            "category": "arena-hard-v0.1",
            "cluster": cluster_number_to_name[str(cluster)],
            "turns": turns_list
        })

    return arena_hard_questions

def to_wandb_table(conversations: List[Dict], image_dir: str) -> wandb.Table:
    data = []
    columns = ["question", "image", "prompt_score"]
    for conv in conversations:
        # conv["conversation_a"][0] is the first turn of the conversation 
        # conv["conversation_a"][0]["content"][1][0] is indexing to the first index of the images
        if isinstance(conv["conversation_a"][0]["content"], list):
            question = conv["conversation_a"][0]["content"][0]

            # Take the first image
            image_hash = conv["conversation_a"][0]["content"][1][0]
            image_path = os.path.join(image_dir, f"{image_hash}.png")
            wandb_image = image_path
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, not included in WANDB.")
                continue
            wandb_image = wandb.Image(image_path)
            data.append([question, wandb_image, conv["prompt_score"]])
        elif isinstance(conv["conversation_a"][0]["content"], str):
            question = conv["conversation_a"][0]["content"]
            data.append([question, conv["prompt_score"]])

    return wandb.Table(data=data, columns=columns)

def main():
    parser = argparse.ArgumentParser(description='Filter prompts based on scores and cluster thresholds.')
    parser.add_argument('--conversations_file', type=str, help='Path to the JSONL file containing conversations')
    parser.add_argument('--clusters_file', type=str, help='Path to the JSON file containing cluster assignments')
    parser.add_argument("--image_dir", type=str, help="Path to the directory containing images")
    parser.add_argument('--prompt_threshold', type=int, default=5, help='Minimum score threshold for individual prompts')
    parser.add_argument('--cluster_threshold', type=int, default=3, help='Minimum average score threshold for clusters')
    parser.add_argument('--output_file', type=str, default='filtered_prompts.json', help='Path to save the filtered prompts')
    parser.add_argument('--wandb_project', type=str, default='arena-hard-auto', help='Wandb project name')
    parser.add_argument("--topics_file", type=str, default="topics.json", help="Path to the file containing topic cluster numbers to names mapping")
    
    args = parser.parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project)
    
    conversations = load_jsonl(args.conversations_file)
    clusters = load_json(args.clusters_file)
    
    filtered_prompts = filter_prompts(conversations, clusters, args.prompt_threshold, args.cluster_threshold)
    
    arena_hard_questions = to_arena_hard_questions_format(filtered_prompts, clusters, args.topics_file, args.image_dir)

    with open(args.output_file, "w") as f:
        for question in arena_hard_questions:
            f.write(json.dumps(question) + "\n")
    
    print(f"Filtered {len(filtered_prompts)} prompts out of {len(conversations)} total.")
    print(f"Results saved to {args.output_file}")

    if args.wandb_project:
        wandb.log({"filtered_prompts": to_wandb_table(filtered_prompts, args.image_dir)})

if __name__ == "__main__":
    main()
