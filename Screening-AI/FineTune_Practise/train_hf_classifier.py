import os
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset , DatasetDict
import torch
import evaluate

from transformers import AutoTokenizer , AutoModelForSequenceClassification , AutoConfig
from transformers import TrainingArguments , Trainer



accuracy = evaluate.load('accuracy')
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}



# Or we can also use 
# checkpoint = "sentence-transformers/all-mpnet-base-v2"

checkpoint = "allenai/longformer-base-4096"





tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)


dataset = load_dataset("csv" , data_files="converted_cleaned_data.csv")



def combine_text(example):
    return {
        "text": f"{tokenizer.cls_token} Resume: {example['Resume']} {tokenizer.sep_token} Job: {example['Job_Description']} {tokenizer.sep_token}",
        "label": example["Decision"]
    }


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=1024
    )

dataset = dataset.map(combine_text)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

split_dataset = tokenized_datasets['train'].train_test_split(test_size=0.2)



tokenized_datasets = DatasetDict({
    "train" : split_dataset['train'],
    "test": split_dataset['test']
})

training_args = TrainingArguments(
    output_dir="./results",
    # num_train_epochs=1,
    max_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    eval_strategy="steps",  
    eval_steps=500
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

print("Best model checkpoint:", trainer.state.best_model_checkpoint)
print("Best metric (accuracy):", trainer.state.best_metric)



eval_results = trainer.evaluate()
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")


# Save the trained model and tokenizer locally
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")