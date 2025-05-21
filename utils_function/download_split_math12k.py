import os
from datasets import load_dataset
import numpy as np

# Create base directory if it doesn't exist
os.makedirs("./data", exist_ok=True)

# Load the dataset
print("Loading dataset from hiyouga/math12k...")
dataset = load_dataset("hiyouga/math12k")

# Get the train split
train_data = dataset["train"]

# Get the total number of rows
total_rows = len(train_data)
print(f"Total rows in train split: {total_rows}")

# Calculate the number of rows per part (should be 1200 as mentioned)
rows_per_part = total_rows // 10
print(f"Rows per part: {rows_per_part}")

# Set random seed for reproducibility
np.random.seed(42)

# Shuffle the dataset indices
indices = np.random.permutation(total_rows)

# Split the indices into 10 parts
for i in range(10):
    # Calculate start and end indices for this part
    start_idx = i * rows_per_part
    end_idx = (i + 1) * rows_per_part if i < 9 else total_rows  # For the last part, go to the end
    
    # Get the indices for this part
    part_indices = indices[start_idx:end_idx]
    
    # Create a new dataset for this part
    part_dataset = train_data.select(part_indices)
    
    # Calculate actual number of samples in this part
    actual_samples = len(part_dataset)
    
    # Create directory structure
    part_dir = f"./data/math12k_part{i+1}"
    train_dir = os.path.join(part_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Save the dataset as parquet
    parquet_path = os.path.join(train_dir, "data.parquet")
    part_dataset.to_parquet(parquet_path)
    
    print(f"Saved part {i+1} with {actual_samples} samples to {parquet_path}")

print("All parts saved successfully!")
print("You can load each part using: load_dataset('parquet', data_files='./data/math12k_part1/train/data.parquet')")
