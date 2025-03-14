import os
import re
import sys
import argparse
import evaluate
import warnings
import numpy as np
import torch.distributed as dist

from dotenv import load_dotenv
from transformers import set_seed
from huggingface_hub import login
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset

from utils.utils import write_to_csv
from utils.templates import Qwen2PromptTemplate
from utils.metrics import build_llm_evaluate_prompt

warnings.filterwarnings("ignore")

set_seed(42)

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--category", required=True,
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--num", required=True)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--output_dir", default="./output")

args = parser.parse_args()
method = args.method
category = args.category
num = args.num

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load('rouge')
meteor_metric = evaluate.load('meteor')

model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
    skip_special_tokens=True,
    temperature=0.8,
    top_p=0.95
)
llm = LLM(model_name, gpu_memory_utilization=0.9)
system_prompt = (
    f"You are an impartial evaluator tasked with assessing the quality of AI-generated personalized item reviews for a specific user. Based on the scoring criteria and the provided item's title and description, along with the user's rating for this item and the review title, evaluate the AI-generated review compared to the user's real review. Be as objective as possible and output only the score.\n\n"
    f"[Scoring Criteria]:\n"
    f"[Score 0]: The AI-generated review is completely different from the user's real review. It does not describe the item correctly and does not reflect the user's thoughts or preferences.\n"
    f"[Score 2]: The AI-generated review has a weak connection to the user's real review. It may mention the item but does not include the key points or personal details from the user's feedback.\n"
    f"[Score 4]: The AI-generated review partially matches the user's real review. Some key points are included, but important details are missing, and the review may feel too general or not personal enough.\n"
    f"[Score 6]: The AI-generated review mostly matches the user's real review. It covers the main points but may miss some details or personal touches that the user included.\n"
    f"[Score 8]: The AI-generated review is very similar to the user's real review. It captures the user's thoughts and preferences well, with only small differences.\n"
    f"[Score 10]: The AI-generated review is almost the same as the user's real review. It includes all key details, personal thoughts, and preferences exactly as the user expressed them.\n"
)
pattern = r"^\s*(\d+\.?\d*)"
pt = Qwen2PromptTemplate(system_prompt)

predictions_path = f"{args.output_dir}/{method}_{category}/predictions_{num}.txt"
with open(predictions_path, 'r', encoding='utf-8') as f:
    predictions = f.read()
    predictions = predictions.split('\n---------------------------------\n')
    predictions = predictions[:-1]

# main_dataset = load_from_disk(f"DPL-main/{category}/{args.dataset}")
# meta_dataset = load_from_disk(f"DPL-meta/{category}/full")
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
datas = [(meta_dataset[sample["data"]["asin"]],
          sample["data"]["rating"],
          sample["data"]["title"],
          sample["data"]["text"])
         for sample in main_dataset]
result_bleu = bleu_metric.compute(predictions=predictions,
                                  references=references)
result_rouge = rouge_metric.compute(predictions=predictions,
                                    references=references)
result_meteor = meteor_metric.compute(predictions=predictions,
                                      references=references)

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
    "rouge-1": result_rouge["rouge1"],
    "rouge-L": result_rouge["rougeL"],
    "bleu": result_bleu["score"],
    "meteor": result_meteor['meteor'],
    "score-72b": score
}

print(f"{args.method} {category} {num} {args.dataset} {result}")

write_to_csv(f"{args.method} {category}", "rouge-1", result["rouge-1"])
write_to_csv(f"{args.method} {category}", "rouge-L", result["rouge-L"])
write_to_csv(f"{args.method} {category}", "bleu", result["bleu"])
write_to_csv(f"{args.method} {category}", "meteor", result["meteor"])
write_to_csv(f"{args.method} {category}", "score-72b", result["score-72b"])

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
