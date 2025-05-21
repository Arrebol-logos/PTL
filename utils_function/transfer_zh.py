import json
from datasets import load_dataset
from tqdm import tqdm

# Load the full training set
ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k", split="train")

# Define target repository names and target score
target_repos = [
    "Haijian/Advanced-Math",
    "gavinluo/applied_math",
    "EduChat-Math"
]
target_score = 10

# Filter samples where repo_name is in the list and score equals 10
math_ds = ds.filter(
    lambda ex: ex.get("repo_name") in target_repos and ex.get("score") == target_score
)
print(f"Number of filtered samples matching criteria: {len(math_ds)}")

# Processing function: concatenate question and answer into a single 'text' field
def process(example):
    question = example["input"]
    answer = example["content"]
    return {"text": f"Question: {question} Answer: {answer}"}

# Apply transformation and remove original columns
math_ds = math_ds.map(process, remove_columns=math_ds.column_names)

# Save as JSONL format
save_path = "R1-Math-Score10.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for ex in tqdm(math_ds, desc="Saving math subset"):
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Math subset has been saved to: {save_path}")