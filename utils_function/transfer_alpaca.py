from datasets import load_dataset
import json
from tqdm import tqdm

# Load dataset
dataset = load_dataset("nvidia/OpenMathInstruct-1", split="train")  # Replace with actual dataset if needed

# Preprocessing function: concatenate question and solution
def process(example):
    return {
        "text": "Question: " + example["question"].strip() + "\nAnswer: " + example["generated_solution"].strip()
    }

# Apply mapping transformation
dataset = dataset.map(process, remove_columns=dataset.column_names)

# Save as JSONL
save_path = "OpenMathInstruct-1.json"
with open(save_path, "w", encoding="utf-8") as f:
    for example in tqdm(dataset, desc="Saving JSONL"):
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"✅ Data saved to: {save_path}")