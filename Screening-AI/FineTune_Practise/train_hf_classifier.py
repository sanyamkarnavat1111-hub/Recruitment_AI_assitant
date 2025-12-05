import os
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset, DatasetDict
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# === 1. Load dataset directly from HF ===
dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

# === 2. Tokenizer & Model ===
checkpoint = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 3 classes!

# === 3. Label mapping (on-the-fly) ===
label2id = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
id2label = {v: k for k, v in label2id.items()}

# Optional: pass to model config for nicer logging
model.config.label2id = label2id
model.config.id2label = id2label

# === 4. Combine text + convert string label → int ===
def prepare_example(example):
    text = f"{tokenizer.cls_token} Resume: {example['resume_text']} {tokenizer.sep_token} Job: {example['job_description_text']} {tokenizer.sep_token}"
    
    label_str = str(example["label"]).strip().lower()
    
    if label_str == "good fit":
        label_id = 2
    elif label_str == "potential fit":
        label_id = 1
    elif label_str == "no fit":
        label_id = 0
    else:
        print(f"WARNING: Unknown label '{label_str}' → treating as No Fit")
        label_id = 0
    
    return {"text": text, "label": label_id}

dataset = dataset.map(
    prepare_example,
    remove_columns=["resume_text", "job_description_text", "label"],  # drop original cols
    desc="Combining text and converting labels"
)

# === 5. Tokenize with global attention on CLS ===
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )
    
    # Global attention: only on the first token (CLS)
    seq_len = len(tokenized["input_ids"][0])
    global_attention_mask = [1] + [0] * (seq_len - 1)  # 1 for CLS, 0 for rest
    tokenized["global_attention_mask"] = [global_attention_mask] * len(examples["text"])
    
    return tokenized

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing with global attention"
)

# === 6. Optional: Create validation set ===
train_val = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
tokenized_datasets = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": tokenized_datasets["test"]
})

# === 7. Metrics ===
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), dim=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1_score}

# === 8. Training ===
training_args = TrainingArguments(
    output_dir="./longformer-resume-3class",
    num_train_epochs=5,
    per_device_train_batch_size=4,        # Adjust based on GPU (8 if 24GB+)
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,        # Effective batch size = 4×4 = 16
    learning_rate=1e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,                            # Remove if on CPU or old GPU
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Final evaluation on test set:")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)