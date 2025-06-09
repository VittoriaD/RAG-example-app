import os
import json

def log_rag_sample(question, answer, contexts, ground_truth=None, path="logs/sessions.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sample = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    if ground_truth:
        sample["ground_truth"] = ground_truth

    with open(path, "a", encoding="utf-8") as f:
        json.dump(sample, f)
        f.write("\n")
