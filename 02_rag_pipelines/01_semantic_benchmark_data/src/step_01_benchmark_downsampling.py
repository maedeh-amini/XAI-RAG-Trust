import pandas as pd
from datasets import load_dataset

# 1. Load the RAGBench HotpotQA test split
print("Loading dataset from Hugging Face...")
dataset = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")

# 2. Shuffle with a fixed seed and select 50 entries
sampled_dataset = dataset.shuffle(seed=42).select(range(50))

# 3. Convert to a pandas DataFrame
df_sampled = sampled_dataset.to_pandas()

# 4. Define the exact columns needed and handle naming variations
columns_to_keep = []

# ID column
if 'id' in df_sampled.columns:
    columns_to_keep.append('id')
elif 'hf-id' in df_sampled.columns:
    df_sampled.rename(columns={'hf-id': 'id'}, inplace=True)
    columns_to_keep.append('id')

# Question column
if 'question' in df_sampled.columns:
    columns_to_keep.append('question')

# Response column
if 'response' in df_sampled.columns:
    columns_to_keep.append('response')
elif 'answer' in df_sampled.columns:
    df_sampled.rename(columns={'answer': 'response'}, inplace=True)
    columns_to_keep.append('response')

# Documents column
if 'documents' in df_sampled.columns:
    columns_to_keep.append('documents')
elif 'contexts' in df_sampled.columns:
    df_sampled.rename(columns={'contexts': 'documents'}, inplace=True)
    columns_to_keep.append('documents')

# Filter the DataFrame to only include these columns
df_final = df_sampled[columns_to_keep]

# 5. Save the standalone IDs for tracking (CSV and TXT)
print("Saving standalone IDs...")
df_final[['id']].to_csv("selected_hf_ids.csv", index=False)
with open("selected_hf_ids.txt", "w") as f:
    f.write("\n".join(df_final['id'].astype(str).tolist()))

# 6. Save the final clean dataset in BOTH formats
print("Saving main datasets...")

# Save as CSV (for visual inspection in Excel)
df_final.to_csv("ragbench_hotpotqa_50_sampled.csv", index=False)

# Save as JSONL (to avoid list-parsing errors)
df_final.to_json("ragbench_hotpotqa_50_sampled.jsonl", orient="records", lines=True)

print(f"\nSuccess! Saved 50 questions.")
print(f"Columns included: {list(df_final.columns)}")
print(f"Outputs generated:")
print(f"  - ragbench_hotpotqa_50_sampled.csv")
print(f"  - ragbench_hotpotqa_50_sampled.jsonl")
print(f"  - selected_hf_ids.csv")
