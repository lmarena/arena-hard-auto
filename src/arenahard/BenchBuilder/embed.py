import pandas as pd
import numpy as np
import pickle
import argparse
import torch

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    print(args)
    
    transformer = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
    
    data = pd.read_json(args.file)
    print(len(data))
    
    ids = data.question_id
    prompts = data.turns.map(lambda x: x[0]["content"])
    
    embeddings = transformer.encode(prompts.tolist(), convert_to_tensor=True, batch_size=8192, show_progress_bar=True)
    torch.save(embeddings, 'embeddings.pt')