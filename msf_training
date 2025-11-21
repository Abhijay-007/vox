!pip install --upgrade transformers
!pip install rouge_score nltk
!pip install -q evaluate
!pip install -q "transformers==4.43.3" "datasets" "accelerate" "evaluate" "sentencepiece"

# Colab cell 3: imports and basic config
import os
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
# Colab cell 4: dataset path -- edit if needed
# This matches the save paths from your script.
possible_paths = [
    "/content/drive/MyDrive/voxlinux_models/metasploit_t5_dataset_500_optionA_randomized.csv",
    "/mnt/data/metasploit_t5_dataset_500_optionA_randomized.csv",
    "/content/metasploit_t5_dataset_500_optionA_randomized.csv",
]
csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(f"Could not find dataset. Checked: {possible_paths}")
print("Using dataset:", csv_path)
raw = load_dataset("csv", data_files=csv_path)["train"]
print("Rows:", len(raw))

# No cast_column calls needed — remove them.
raw = raw.select(range(len(raw)))  # optional: force materialization

raw[0]
# Colab cell 6: model + tokenizer
MODEL_NAME = "t5-base"   # swap to "t5-small" to experiment faster
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
# Colab cell 7: preprocessing / tokenization
# Trainer hyperparameters (adjust for your GPU)
max_input_length = 128
max_target_length = 128

def preprocess(batch):
    # batch: dict with 'input_text' and 'target_text' lists
    inputs = tokenizer(batch["input_text"], max_length=max_input_length, truncation=True, padding="max_length")
    targets = tokenizer(batch["target_text"], max_length=max_target_length, truncation=True, padding="max_length")
    inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in target_ids]
        for target_ids in targets["input_ids"]
    ]
    return inputs

tokenized = raw.map(preprocess, batched=True, remove_columns=raw.column_names)
tokenized = tokenized.train_test_split(test_size=0.05, seed=42)  # small val split
print(tokenized)
# Colab cell 8: Data collator and metrics
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 in the labels as pad_token_id
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # basic ROUGE (use rougeL and Rouge1/2)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # BLEU (tokenized)
    bleu_result = {}
    try:
        bleu_result = bleu.compute(predictions=[p.split() for p in decoded_preds],
                                   references=[[r.split()] for r in decoded_labels])
    except Exception:
        bleu_result = {"bleu": 0.0}

    # return a combined dict with some common metrics
    result = {
        "rouge1": rouge_result.get("rouge1", 0.0),
        "rouge2": rouge_result.get("rouge2", 0.0),
        "rougeL": rouge_result.get("rougeL", 0.0),
        "bleu": bleu_result.get("bleu", 0.0),
    }
    return result
# Colab cell 9: Training arguments (Trainer tokens / hyperparams)
output_dir = "/content/drive/MyDrive/voxlinux_models/t5_metasploit_base"  # save to Drive
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_steps=50,
    save_total_limit=3,
    seed=42,
    # Remove evaluation_strategy, save_strategy, fp16, load_best_model_at_end, metric_for_best_model
)
# Colab cell 10: Trainer init and train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
# Colab cell 11: save & sample generation
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# quick generation demo - sample a few rows
examples = raw.select(range(min(8, len(raw))))["input_text"]
for ex in examples:
    inputs = tokenizer(ex, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
    gen = model.generate(**inputs, max_length=128, num_beams=4)
    print("INPUT:", ex)
    print("OUTPUT:", tokenizer.decode(gen[0], skip_special_tokens=True))
    print("-" * 60)
