import os
import random
from tqdm import tqdm
from collections import Counter
from pathlib import Path

def maximize_datasets(input_paths, output_dir, output_prefix, chunk_size=100_000):
    """
    Merge multiple (or single) datasets, shuffle and split:
      - If a single dataset is given, only print total line count
      - If multiple datasets, print per-chunk statistics for each source
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory recursively if it doesn't exist

    # 1) Read and label
    all_records = []  # Store (line, src_idx) tuples
    for src_idx, path in enumerate(input_paths):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        all_records += [(line, src_idx) for line in lines]
        print(f"Read {path} with {len(lines)} lines")

    total = len(all_records)
    max_datasets = total // chunk_size
    if max_datasets == 0:
        raise ValueError(f"Not enough data! At least {chunk_size} lines required, only {total} found.")

    print(f"Total data: {total} lines | Can generate: {max_datasets} datasets | Each with: {chunk_size} lines")

    # 2) Global shuffle (in-place)
    random.shuffle(all_records)

    # 3) Chunk and save
    single_source = (len(input_paths) == 1)
    names = [Path(p).stem for p in input_paths]

    for i in tqdm(range(max_datasets), desc="Generation Progress"):
        chunk = all_records[i*chunk_size : (i+1)*chunk_size]
        output_path = os.path.join(output_dir, f"{output_prefix}_{i+1}.jsonl")

        # 3a) Print statistics
        if single_source:
            print(f"Chunk {i+1}: Total {len(chunk)} lines")
        else:
            counts = Counter(src for _, src in chunk)  # Count how many lines from each source
            parts = [
                f"{names[src]}: {counts.get(src, 0)}"
                for src in range(len(input_paths))
            ]
            print(f"Chunk {i+1}: " + " | ".join(parts))

        # 3b) Write to file
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line, _ in chunk:
                fout.write(line)

    # 4) Handle remaining lines
    remaining = total % chunk_size
    if remaining:
        remain_chunk = all_records[-remaining:]
        remain_path = os.path.join(output_dir, f"{output_prefix}_remaining_{remaining}_lines.jsonl")

        if single_source:
            print(f"{remaining} remaining lines, saved to {remain_path}")
        else:
            counts = Counter(src for _, src in remain_chunk)
            parts = [
                f"{names[src]}: {counts.get(src, 0)}"
                for src in range(len(input_paths))
            ]
            print(f"{remaining} remaining lines ({' | '.join(parts)}), saved to {remain_path}")

        with open(remain_path, 'w', encoding='utf-8') as fout:
            for line, _ in remain_chunk:
                fout.write(line)

if __name__ == "__main__":
    maximize_datasets(
        input_paths=["./data/NuminaMath-CoT.jsonl",
                     "./data/MetaMathQA.jsonl"],
        output_dir="./data",
        output_prefix="math_cot",
        chunk_size=100000
    )
