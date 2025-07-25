import os
import sys
import pickle
import argparse
import warnings
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import set_seed, AutoTokenizer

from utils.utils import postprocess_output
from utils.templates import Qwen2PromptTemplate
from utils.preprocess import create_prompt_generator, GeneralDataset

warnings.filterwarnings("ignore")

set_seed(42)
if dist.is_initialized():
    dist.barrier()

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--category", required=True,
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--output_dir", required=True)
parser.add_argument("--max_tokens", type=int, default=4096)
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
        os.makedirs(f"{output_dir}/{method}_{category}", exist_ok=True)

    if os.path.exists(f"{output_dir}/{method}_{category}/diff_inputs_{category}.pkl"):
        with open(f"{output_dir}/{method}_{category}/diff_inputs_{category}.pkl", "rb") as f:
            diff_inputs = pickle.load(f)
    else:
        diff_inputs = None

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    prompt_generator = create_prompt_generator(
        num_retrieved=num,
        user_profile_map=user_profile_map,
        asin_reviewers_map=asin_reviewers_map,
        embedder=None,
    )

    dataset = GeneralDataset(
        main_dataset=main_dataset,
        diff_inputs=diff_inputs,
        user_profile_map=user_profile_map,
        asin_map=asin_map,
        prompt_generator=prompt_generator
    )
    dataset = [(inp_creator, summ_creator, diff_inp, out)
               for inp_creator, summ_creator, diff_inp, out
               in tqdm(dataset, desc="Data-processing", total=len(dataset))]
    inp_creators, summ_creators, diff_inputs, references = zip(*dataset)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        temperature=args.temperature,
        top_p=0.95
    )
    llm = LLM(
        model_name,
        tensor_parallel_size=len(args.gpu.split(",")),
        gpu_memory_utilization=0.8
    )

    diff_system_prompt = (
        f"Given the title and description of an item, along with the current user's review and 4 other users' reviews for the same item, "
        f"analyze and output the differences between the current user and other users by considering the following aspects:\n"
        f"[Writing Style]: word choice and sentence structure.\n"
        f"[Emotional Style]: sentiment tone (positive, negative, or neutral).\n"
        f"[Semantic Style]: information density and contextual coherence.\n"
    )
    diff_pt = Qwen2PromptTemplate(
        system_prompt=diff_system_prompt
    )

    differences = [[] for _ in range(len(dataset))]
    for i in range(num):
        diff_input = [diff_pt.build_prompt(diff_inp[i])
                      for diff_inp in diff_inputs]
        diff = llm.generate(diff_input, sampling_params)
        diff = [d.outputs[0].text.strip() for d in diff]
        for j, d in enumerate(diff):
            differences[j].append(d)

    item_intro = 'an item' if num == 1 else f'{num} items'
    summarizer_system_prompt = (
        f"Given titles and descriptions of {item_intro}, along with the differences in writing style, emotional style, and semantic style between the current user's review and other users' reviews for each item, and the current user's past reviews, "
        f"generate a profile summary of the current user.\n"
        f"The summary should be formatted as follows:\n"
        f"[Summary]: <summary>"
    )
    summarizer_pt = Qwen2PromptTemplate(
        system_prompt=summarizer_system_prompt
    )

    summ_inputs = [summ_creator(diff)
                   for diff, summ_creator
                   in zip(differences, summ_creators)]
    summ_inputs = [summarizer_pt.build_prompt(summ_inp)
                   for summ_inp in summ_inputs]

    summaries = llm.generate(summ_inputs, sampling_params)
    summaries = [s.outputs[0].text.strip() for s in summaries]

    generator_system_prompt = (
        f"Given the title and description of an item, along with the current user's past reviews and profile summary, and the output review rating and review title, "
        f"generate a personalized item review for the current user.\n"
        f"The review should be formatted as follows:\n"
        f"[Review]: <review>"
    )
    generator_pt = Qwen2PromptTemplate(
        system_prompt=generator_system_prompt
    )

    inputs = [inp_creator(summ)
              for summ, inp_creator
              in zip(summaries, inp_creators)]
    inputs = [generator_pt.build_prompt(inp) for inp in inputs]

    predictions = llm.generate(inputs, sampling_params)
    predictions = [postprocess_output(prediction.outputs[0].text)
                   for prediction in tqdm(predictions, desc="Post-processing", total=len(predictions))]

    with open(f"{output_dir}/{method}_{category}/predictions_{num}.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n---------------------------------\n")

    print(args)
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
