import json
from datasets import load_dataset
from tqdm import tqdm

# Load dataset
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
model_save_path = 'NuminaMath-CoT.jsonl'

# Concatenate 'query' and 'response' fields into a single 'text' field
def process(example):
    return {
        "text": "Question: " + example["messages"][0]['content'] + 
                "Answer: " + example["messages"][1]['content']
    }

# Apply transformation and remove original fields
dataset = dataset.map(process, remove_columns=dataset.column_names)

# Print the first 10 samples for inspection
for i, example in enumerate(dataset.select(range(10)), start=1):
    print(f"Sample {i}:")
    print(example)
    print("=" * 40)

# Save as JSONL format (one JSON object per line)
with open(model_save_path, "w", encoding="utf-8") as f:
    for example in tqdm(dataset, desc="Saving as JSONL"):
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"✅ Data successfully saved to {model_save_path}")


