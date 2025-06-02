import re
import sys
import argparse
import warnings
import numpy as np
import torch.distributed as dist

from datasets import load_dataset
from transformers import set_seed
from vllm import LLM, SamplingParams

from utils.templates import Qwen2PromptTemplate
from utils.metrics import build_llm_evaluate_prompt
from utils.utils import write_to_csv, postprocess_output

warnings.filterwarnings("ignore")

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--category", required=True,
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--num", required=True, type=int)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--output_dir", default="./output")

args = parser.parse_args()
method = args.method
category = args.category
output_dir = args.output_dir
num = args.num

predictions_path = f"{output_dir}/{method}_{category}/predictions_{num}.txt"
references_path = f"{output_dir}/{method}_{category}/references_{num}.txt"
with open(predictions_path, 'r', encoding='utf-8') as f:
    predictions = f.read()
    predictions = predictions.split('\n---------------------------------\n')
    predictions = predictions[:-1]
predictions = [postprocess_output(pred) for pred in predictions]

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
meta_dataset = dict(zip(meta_dataset["asin"],
                        zip(meta_dataset["title"],
                            meta_dataset["description"])))
references = [sample["data"]["text"] for sample in main_dataset]
datas = [(meta_dataset[sample["data"]["asin"]][0],
          meta_dataset[sample["data"]["asin"]][1],
          sample["data"]["rating"], 
          sample["data"]["title"])
        for sample in main_dataset]

model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
sampling_params = SamplingParams(
    max_tokens=64,
    skip_special_tokens=True,
    temperature=0.8,
    top_p=0.95
)
llm = LLM(model_name, gpu_memory_utilization=0.9)
system_prompt = (
    f"You are an impartial evaluator tasked with assessing how well an AI-generated item review is personalized for a specific user. Based on the provided item title and description, the user's rating for this item, and the review title, assign a score to the AI-generated review by comparing it to the user's real review."
    f"[Scoring Criteria]:\n"
    f"[Score 0]: The AI-generated review is completely unrelated to the user's real review. It does not describe the item correctly or reflects the user's personal thoughts and preferences.\n"
    f"[Score 2]: The AI-generated review has a weak connection to the user's real review. It may mention the item but does not include the key points or personal thoughts that the user included.\n"
    f"[Score 4]: The AI-generated review partially matches the user's real review. Some key points are included, but important details are missing, and the review may feel too general or not personal enough.\n"
    f"[Score 6]: The AI-generated review mostly matches the user's real review. It covers the main points but may miss some details or personal thoughts that the user included.\n"
    f"[Score 8]: The AI-generated review is very similar to the user's real review. It captures the user's thoughts and preferences well, with only small differences.\n"
    f"[Score 10]: The AI-generated review is almost the same as the user's real review. It includes all key details, personal thoughts, and preferences exactly as the user expressed them.\n"
    f"Output the numerical score only."
)
pattern = r"^\s*(\d+\.?\d*)"
pt = Qwen2PromptTemplate(system_prompt)

prompts = [pt.build_prompt(build_llm_evaluate_prompt(pd, gt, data))
        for pd, gt, data in zip(predictions, references, datas)]
scores = llm.generate(prompts, sampling_params)
scores = [int(re.match(pattern, score.outputs[0].text).group(1))
        for score in scores]
for score in scores:
    if score < 0 or score > 10:
        raise ValueError("Invalid score value.")
score = np.mean(scores) / 10

result = {
    "score-72B": score
}

print(f"{args.method} {category} {args.dataset} {result}")

write_to_csv(f"{args.method} {category} {num}", "score-72B", result["score-72B"])

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
