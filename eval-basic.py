import sys
import argparse
import evaluate
import warnings
import torch.distributed as dist

from datasets import load_dataset
from transformers import set_seed

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
references = [sample["data"]["text"] for sample in main_dataset]

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load('rouge')
meteor_metric = evaluate.load('meteor')
result_bleu = bleu_metric.compute(predictions=predictions,
                                  references=references)
result_rouge = rouge_metric.compute(predictions=predictions,
                                    references=references)
result_meteor = meteor_metric.compute(predictions=predictions,
                                      references=references)

result = {
    "rouge-1": result_rouge["rouge1"],
    "rouge-L": result_rouge["rougeL"],
    "meteor": result_meteor['meteor'],
    "bleu": result_bleu["score"],
}
print(result)

write_to_csv(f"{args.method} {category} {num}", "rouge-1", result["rouge-1"])
write_to_csv(f"{args.method} {category} {num}", "rouge-L", result["rouge-L"])
write_to_csv(f"{args.method} {category} {num}", "meteor", result["meteor"])
write_to_csv(f"{args.method} {category} {num}", "bleu", result["bleu"])

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
