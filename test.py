import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import evaluate
import numpy as np

# === Paths ===
MODEL_NAME = "meta-llama/Llama-3.2-3B"
CHECKPOINT_PATH = "output/llama-3.2-3b-alpaca-lora/checkpoint-2000"
TEST_CSV = "test_data.csv"

# === Load tokenizer and model with LoRA ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading LoRA adapter from {CHECKPOINT_PATH}...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()
torch.cuda.empty_cache()

# === Load test set (subset for evaluation) ===
print(f"Loading test set from {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV).sample(n=200, random_state=42)

# === Prompt formatting and tokenization ===
def format_instruction(example):
    instruction = example["instruction"] or ""
    context = example["context"] if pd.notna(example["context"]) else ""
    response = example["response"] or ""
    if context:
        prompt = f"[INST] {instruction} \n\n {context} [/INST] {response}"
    else:
        prompt = f"[INST] {instruction} [/INST] {response}"
    return {"formatted_text": prompt}

def tokenize_function(examples):
    results = tokenizer(
        examples["formatted_text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    results["labels"] = results["input_ids"].copy()
    return results

# === Prepare test dataset ===
print("Formatting and tokenizing...")
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(format_instruction)
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

# === Define evaluation metrics ===
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = [p.strip() for p in predictions]
    labels = [l.strip() for l in labels]

    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=labels)

    bleu_scores = [
        sentence_bleu([word_tokenize(label)], word_tokenize(pred))
        for pred, label in zip(predictions, labels)
    ]
    avg_bleu = np.mean(bleu_scores)

    bertscore = evaluate.load("bertscore")
    bertscore_result = bertscore.compute(predictions=predictions, references=labels, lang="en")

    return {
        "bleu": avg_bleu,
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bertscore_f1": np.mean(bertscore_result["f1"]),
    }

# === Manual inference loop ===
print("Running manual evaluation...")
all_predictions = []
all_labels = []

for i, sample in enumerate(test_dataset):
    input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_label = tokenizer.decode(sample["labels"], skip_special_tokens=True)
    all_predictions.append(decoded_pred)
    all_labels.append(decoded_label)

    if (i + 1) % 25 == 0:
        print(f"✓ Evaluated {i + 1} samples...")

# === Compute and print metrics ===
metrics = compute_metrics((all_predictions, all_labels))

print("\n=== Evaluation Results ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
