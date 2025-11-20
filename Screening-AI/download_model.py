import os
os.environ["HF_HOME"] = "D:/Recruitment_AI_assitant/Screening-AI/Hugging_Face"

# The order of "import" for setting environment variable is very crucial 
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_TOKEN'] = "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"

checkpoint = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)


print("Done ...")