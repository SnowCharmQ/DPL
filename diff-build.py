import os
import pickle
import argparse
import warnings

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import set_seed
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from utils.preprocess import create_prompt_generator, GeneralDataset

warnings.filterwarnings("ignore")

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--category", required=True,
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--output_dir", required=True)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--num_retrieved", type=int,
                    required=True, choices=range(1, 9))
parser.add_argument("--retriever", default="bm25")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--gpu")

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == "__main__":
    print(Path(__file__).resolve())
    with open(__file__, 'r', encoding='utf-8') as f:
        content = f.read()
    output_dir = args.output_dir
    method = args.method
    num = args.num_retrieved
    category = args.category

    if not os.path.exists(f"{output_dir}/{method}_{category}"):
        os.makedirs(f"{output_dir}/{method}_{category}")

    main_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split=args.dataset
    )
    meta_dataset = load_dataset(
        "SnowCharmQ/DPL-meta",
        category,
        split="full"
    )

    user_profile_map = {}
    asin_reviewers_map = defaultdict(set)
    for sample in main_dataset:
        user_id = sample["user_id"]
        user_profile_map[user_id] = sample["profile"]
        for p in sample["profile"]:
            asin_reviewers_map[p["asin"]].add(user_id)
    asin_map = dict(zip(meta_dataset["asin"], 
                        zip(meta_dataset["title"], 
                            meta_dataset["description"])))

    embedder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    prompt_generator = create_prompt_generator(
        num_retrieved=num,
        user_profile_map=user_profile_map,
        asin_reviewers_map=asin_reviewers_map,
        embedder=embedder,
    )
    
    dataset = GeneralDataset(
        main_dataset=main_dataset, 
        diff_inputs=None,
        user_profile_map=user_profile_map,
        asin_map=asin_map, 
        prompt_generator=prompt_generator
    )
    dataset = [diff_inp
               for _, _, diff_inp, _
               in tqdm(dataset, desc="Data-processing", total=len(dataset))]
    
    with open(f"{output_dir}/{method}_{category}/diff_inputs_{category}.pkl", "wb") as f:
        pickle.dump(dataset, f)
