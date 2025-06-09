from datasets import load_dataset
from ragas import evaluate
from collections import defaultdict
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)

from dotenv import load_dotenv
load_dotenv()

dataset = load_dataset(
    "json",
    data_files="logs/sessions.jsonl"
)["train"]


result = evaluate(
    dataset,
    metrics=[answer_relevancy, faithfulness]
)


metric_totals = defaultdict(list)
for sample in result.scores:
    for metric_name, score in sample.items():
        metric_totals[metric_name].append(float(score))

print("Evaluation Results:")
for metric_name, values in metric_totals.items():
    avg = sum(values) / len(values)
    print(f"{metric_name}: {avg:.4f}")
