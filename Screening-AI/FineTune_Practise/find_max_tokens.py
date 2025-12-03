import os
from dotenv import load_dotenv

load_dotenv()


from transformers import AutoTokenizer
import pandas as pd

# Load your tokenizer
checkpoint = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load your dataset
df = pd.read_csv("converted_cleaned_data.csv")

def get_token_length(row):
    text = f"Resume: {row['Resume']} [SEP] Job: {row['Job_Description']}"
    tokens = tokenizer.encode(text, truncation=False)
    return len(tokens)

df["token_length"] = df.apply(get_token_length, axis=1)

print("Max token length:", df["token_length"].max())
print("Average token length:", df["token_length"].mean())
print("95th percentile:", df["token_length"].quantile(0.95))


