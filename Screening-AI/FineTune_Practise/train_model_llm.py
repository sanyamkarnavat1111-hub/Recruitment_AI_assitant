
import os 

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"
from dotenv import load_dotenv

load_dotenv()

import torch

# 2. Force CUDA memory to report correctly (backup)
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        _ = torch.tensor([0], device='cuda')  # Force memory query to work
    except:
        pass

print("Unsloth fused loss disabled + CUDA fixed. Training will work now.")
# ==================================

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig , SFTTrainer





os.environ['HF_HOME'] = "/mnt/d/Recruitment_AI_assitant/Screening-AI/Hugging_Face/"
# os.environ['HF_TOKEN'] = "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"

max_seq_length = 2048 # Can keep any uses Rope scaling internally
dtype = None # Auto detects if the dtype is kept none
load_in_4bit = True


dataset = load_dataset("csv", data_files="cleaned_data.csv")


print(f"Dataset loaded ... {dataset}")


# All models available from unsloth 

fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",  # Gemma 2.2x faster!
]  # More models at https://huggingface.co/unsloth


model , tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)


#  LoRA adapters so we only need to update 1 to 10% of all parameters!


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

prompt = """Below is the provided resume information of the candidate and also the job description for which the candidate 
applied for along with the information that if the candidate is fit or unfit for the given job description (0 denotes unfit , 1 denotes fit)

Your job as a human resource candidate resume and job description matcher to evaluate if the candidate is fit or unfit.
Also use your own logical reasoning as well to predict and contribute to the decision 
### Resume Text :
{}

### Job description:
{}

### Decision
{}
"""
EOS_TOKEN = tokenizer.eos_token

def format_prompt_func(examples):
    # Extract lists of column values
    Resume_Text = examples['Resume']
    Job_Description = examples['Job_Description']
    decision = examples['Decision']
    
    # Prepare list of processed texts
    texts = []
    for resume, job_desc, desc in zip(Resume_Text, Job_Description, decision):
        text = prompt.format(resume, job_desc, desc) + EOS_TOKEN
        texts.append(text)
        
    return texts

train_dataset = dataset['train']


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    formatting_func=format_prompt_func,
    args=SFTConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        warmup_steps=5,
        max_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="./Result/outputs",
    )
)
trainer_stats = trainer.train()


print("Training done ...")


model.save_pretrained_merged("model_4_bit", tokenizer, save_method = "merged_4bit_forced",)

print("4 bit model saved ")

# Save model in 8bit precision
model.save_pretrained_merged(
    "pretrained_model",  # folder where model will be saved
    tokenizer,
    save_method="merged_16bit"
)

print("8 bit model saved ...")