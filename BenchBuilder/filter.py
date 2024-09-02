"""
Filter prompts based on scores and cluster thresholds. To be run after topic_clustering.py and label.py
"""

import orjson
import json
import argparse
from typing import List, Dict
import numpy as np

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
    
    return {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}

def filter_prompts(conversations: List[Dict], clusters: List[int], prompt_threshold: int, cluster_threshold: float) -> List[Dict]:
    cluster_scores = calculate_cluster_scores(conversations, clusters)
    
    filtered_prompts = []
    for conv, cluster in zip(conversations, clusters):
        score = calculate_score(conv)
        if score >= prompt_threshold and cluster_scores[cluster] >= cluster_threshold:
            filtered_prompts.append(conv)
    
    return filtered_prompts

def main():
    parser = argparse.ArgumentParser(description='Filter prompts based on scores and cluster thresholds.')
    parser.add_argument('--conversations_file', type=str, help='Path to the JSONL file containing conversations')
    parser.add_argument('--clusters_file', type=str, help='Path to the JSON file containing cluster assignments')
    parser.add_argument('--prompt_threshold', type=int, default=5, help='Minimum score threshold for individual prompts')
    parser.add_argument('--cluster_threshold', type=int, default=3, help='Minimum average score threshold for clusters')
    parser.add_argument('--output_file', type=str, default='filtered_prompts.json', help='Path to save the filtered prompts')
    
    args = parser.parse_args()
    
    conversations = load_jsonl(args.conversations_file)
    clusters = load_json(args.clusters_file)
    
    filtered_prompts = filter_prompts(conversations, clusters, args.prompt_threshold, args.cluster_threshold)
    
    with open(args.output_file, 'w') as f:
        json.dump(filtered_prompts, f, indent=2)
    
    print(f"Filtered {len(filtered_prompts)} prompts out of {len(conversations)} total.")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
