import os
import re
import openai
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import set_seed
from huggingface_hub import login
from datasets import load_from_disk, load_dataset
from concurrent.futures import as_completed, ThreadPoolExecutor

from utils.metrics import build_llm_evaluate_prompt

warnings.filterwarnings("ignore")

set_seed(42)

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def evaluate_prompt(prompt, system_prompt, client):
    try:
        model = "gpt-4o-mini"
        temperature = 0.8
        max_tokens = args.max_tokens
        top_p = 0.95

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in processing prompt: {e}")
        return "0"


def extract_score(response):
    pattern = r"^\s*(\d+\.?\d*)"
    match = re.match(pattern, response)
    return int(match.group(1)) if match else 0


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
num = args.num
category = args.category

predictions_path = f"{args.output_dir}/{method}_{category}/predictions_{num}.txt"
with open(predictions_path, 'r', encoding='utf-8') as f:
    predictions = f.read()
    predictions = predictions.split('\n---------------------------------\n')
    predictions = predictions[:-1]

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

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                       base_url=os.getenv("OPENAI_BASE_URL"))
prompts = [build_llm_evaluate_prompt(pd, gt, data)
           for pd, gt, data in zip(predictions, references, datas)]
results = []
with ThreadPoolExecutor(max_workers=20) as executor:
    future_to_prompt = {
        executor.submit(evaluate_prompt, prompt, system_prompt, client): prompt
        for prompt in prompts
    }
    for future in tqdm(as_completed(future_to_prompt), total=len(future_to_prompt), desc="Evaluating responses"):
        response = future.result()
        results.append(response)

scores = [extract_score(response) for response in results]
if scores and len(scores) == len(prompts):
    score = np.mean(scores) / 10
    print("\nAverage score:", score)
