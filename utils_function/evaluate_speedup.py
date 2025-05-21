import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # Import datasets library
import time
import tqdm  # Use tqdm for progress bar

# --- 1. Load Model and Tokenizer ---
model_identifier = "google/gemma-2-9b"

tokenizer = AutoTokenizer.from_pretrained(model_identifier)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Note: LLaMA/Gemma models often use bfloat16. If supported, it's recommended.
try:
    model = AutoModelForCausalLM.from_pretrained(model_identifier, torch_dtype=torch.bfloat16).to(device)
    print("Model loaded with bfloat16.")
except (TypeError, ValueError) as e:
    print(f"Could not load with bfloat16 ({e}). Loading with default dtype (likely float32).")
    model = AutoModelForCausalLM.from_pretrained(model_identifier).to(device)

model.eval()

# --- 2. Load GSM8K Test Dataset ---
print("Loading dataset...")
try:
    # Try loading directly from Hugging Face
    dataset = load_dataset("knoveleng/Minerva-Math", split="train")
    print("Dataset loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Could not load dataset from Hugging Face: {e}")
    print("Please ensure you have internet access or load from a local path if available.")

# --- 3. Configure Testing Parameters ---
num_data_points_to_sample = 10  # Sample 10 data points from dataset
# Note: For large models like LLaMA, max_new_tokens has significant impact.
# Choose a representative generation length and keep it consistent across comparisons.
max_new_tokens = 64  # <-- Adjust based on real use case and keep consistent

if len(dataset) < num_data_points_to_sample:
    print(f"Warning: Dataset has only {len(dataset)} samples, testing on all available samples.")
    num_data_points_to_sample = len(dataset)

# --- 4. Measure Inference Time and Compute Average ---
total_inference_time = 0

# Optional: Warmup (process a few samples before timing)
print(f"\nStarting warmup runs with the first few GSM8K data points...")
warmup_samples = min(5, num_data_points_to_sample)  # Warm up 5 samples or fewer
if warmup_samples > 0:
    for i in range(warmup_samples):
        if i >= len(dataset):  # Prevent index out of range
            break
        sample = dataset[i]
        input_text = sample['problem']  # Use 'problem' field in GSM8K

        # Tokenize input (batch size = 1)
        # truncation is important to avoid overly long inputs
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        if device.type == 'cuda':
            torch.cuda.synchronize()
    print("Warmup finished.")
else:
    print("Skipping warmup as num_data_points_to_sample is too small.")

print(f"\n--- Measuring Inference Time on {num_data_points_to_sample} GSM8K test samples ---")
# Iterate over the selected number of data points and measure inference time
start_index_for_timing = warmup_samples

for i in tqdm.tqdm(range(start_index_for_timing, num_data_points_to_sample), desc="Measuring Inference"):
    if i >= len(dataset):
        print(f"Warning: Reached end of dataset at index {i}. Stopping measurement.")
        break

    sample = dataset[i]
    input_text = sample['problem']  # Use 'problem' field in GSM8K

    # Tokenize input (batch size = 1)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    start_time = time.time()
    with torch.no_grad():
        output_sequences = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ensure GPU ops are complete

    end_time = time.time()
    duration = end_time - start_time

    total_inference_time += duration
    # print(f"  Data point {i+1}: {duration:.4f} seconds")  # Uncomment to print individual times

# --- 5. Compute Average Time ---
actual_timed_samples_count = num_data_points_to_sample - start_index_for_timing
if actual_timed_samples_count > 0:
    average_time_per_sample = total_inference_time / actual_timed_samples_count
    print(f"\nTotal inference time for {actual_timed_samples_count} timed samples: {total_inference_time:.4f} seconds")
    print(f"Average inference time per sample: {average_time_per_sample:.4f} seconds")
else:
    print("\nNo samples were processed for timing after warmup.")