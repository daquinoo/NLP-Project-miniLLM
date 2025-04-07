import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from evaluate import load
import evaluate
from transformers import EarlyStoppingCallback, TrainerCallback
from peft import PeftModel
# from trl import DPOTrainer  # For Direct Preference Optimization

# Load preprocessed data
train_df = pd.read_csv("train_data.csv")  
test_df = pd.read_csv("test_data.csv")

# ---------- ADDITIONAL DATA PREPARATION FOR DPO ----------
# Here you would prepare paired preference data for DPO training
# This would include positive (preferred) and negative (rejected) responses for the same prompt
# Example:
# def prepare_dpo_data(train_df):
#     # Format data for DPO training with chosen/rejected responses
#     return dpo_train_dataset, dpo_eval_dataset

# Model initialization
MODEL_NAME = "meta-llama/Llama-3.2-3B"  # Adjust if needed for specific  model path
OUTPUT_DIR = "output/llama-3.2-3b-alpaca-lora"

# LoRA parameters
lora_config = LoraConfig(
    r=16,                      # Rank - smaller = more parameter efficient, larger = more capacity
    lora_alpha=32,             # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers to apply LoRA
    lora_dropout=0.05,         # Dropout probability
    bias="none",               # Don't train bias params
    task_type=TaskType.CAUSAL_LM  # Fine-tuning causal LM
)

# Tokenizer and special tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be EOS token
tokenizer.padding_side = "right"           # Set padding to happen on the right

# Model prep
def prepare_model():
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,      # Half-precision
        device_map="auto",              # Auto-dist across available GPUs
    )
    
    # Prepare model for LoRA fine-tuning
    print("Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model

# Data formatting 
def format_instruction(example):
    """Format inputs as instruction-following samples."""
    instruction = example["instruction"] or ""
    context = example["context"] if example["context"] and not pd.isna(example["context"]) else ""
    response = example["response"] or ""
    
    # Format as: <s>[INST] {instruction} [/INST] {response} </s>
    # If context provided: <s>[INST] {instruction} \n\n {context} [/INST] {response} </s>
    if context:
        prompt = f"[INST] {instruction} \n\n {context} [/INST] {response}"
    else:
        prompt = f"[INST] {instruction} [/INST] {response}"
    
    return {"formatted_text": prompt}

# Tokenizer
def tokenize_function(examples):
    """Tokenize the texts and prepare for training."""
    # Tokenize, truncate and pad sequences
    results = tokenizer(
        examples["formatted_text"],
        truncation=True,
        max_length=512,  # Adjust as needed based on your data
        padding="max_length"
    )
    
    # Create labels by copying input_ids
    results["labels"] = results["input_ids"].copy()
    
    return results

# Prepare dataset
def prepare_datasets(train_df, test_df):
    # Convert dataframes to HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Apply formatting
    train_dataset = train_dataset.map(format_instruction)
    test_dataset = test_dataset.map(format_instruction)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=test_dataset.column_names
    )
    
    return train_dataset, test_dataset

# ---------- EVAL METRICS SETUP ----------
# This is where you would implement the evaluation metrics (SJ)
# def compute_metrics(eval_preds):
#     """
#     Compute ROUGE, BERTScore, and perplexity.
#     """
      #Implement here
#     ...
#     
#     metrics = {
#         "rouge1": rouge_result["rouge1"],
#         "rouge2": rouge_result["rouge2"],
#         "rougeL": rouge_result["rougeL"],
#         "bertscore_f1": np.mean(bertscore_result["f1"]),
#         # "perplexity": perplexity
#     }
#     
#     return metrics

# ---------- INFERENCE CALLBACK FOR VALIDATION EXAMPLES ----------
# Implement a callback for periodic inference on held-out examples
# class InferenceCallback(TrainerCallback):
#     """
#     Callback to run inference on a few examples during training
#     and log the results to tensorboard/wandb.
#     """
#     def __init__(self, validation_examples, tokenizer, log_steps=1000):
#         self.validation_examples = validation_examples
#         self.tokenizer = tokenizer
#         self.log_steps = log_steps
#         
#     ...
#             
#     return control

# ---------- HUMAN EVALUATION INTERFACE SETUP ----------
# Would typically be a separate script

# Training args
def get_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,             # Number of training epochs
        per_device_train_batch_size=4,  # Batch size per GPU for training
        per_device_eval_batch_size=4,   # Batch size per GPU for evaluation
        gradient_accumulation_steps=8,  # Number of steps to accumulate gradients (effective batch size = batch_size * gradient_accumulation_steps)
        evaluation_strategy="steps",    # Evaluate during training
        eval_steps=500,                 # Evaluate every 500 steps
        save_strategy="steps",          # Save during training
        save_steps=500,                 # Save every 500 steps
        save_total_limit=3,             # Keep only the 3 most recent checkpoint
        # ---------- LEARNING RATE MODIFICATION (Nilay) ----------
        # learning_rate=2e-4,           # Learning rate - typically higher for LoRA fine-tuning, change from default (2e-4) to whichever is preferred (1e-5 to 5e-5, or otherwise)
        weight_decay=0.01,              # Weight decay for regularization
        warmup_ratio=0.03,              # Percentage of steps for warmup
        lr_scheduler_type="cosine",     # Learning rate scheduler
        logging_steps=100,              # Log every 100 steps
        fp16=True,                      # Use mixed precision training
        bf16=False,                     # Don't use bfloat16
        report_to="tensorboard",        # Report to TensorBoard
        remove_unused_columns=False,    # Keep all columns
        push_to_hub=False,              # Don't push to HuggingFace Hub
        load_best_model_at_end=True,    # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss as the metric for selecting the best model
        greater_is_better=False,        # Lower loss is better
        # ---------- EARLY STOPPING IMPLEMENTATION (Nilay) ----------
        # Add early stopping parameters, examples below
        # early_stopping_patience=3,    # Default stop after 3 evaluations with no improvement
        # early_stopping_threshold=0.01,  # Default min improvement needed to consider as improvement
    )

# Main training function
def train():
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(train_df, test_df)
    
    # Prepare model
    model = prepare_model()
    
    # Get training arguments
    training_args = get_training_args()
    
    # ---------- CALLBACKS FOR MONITORING AND EARLY STOPPING ----------
    # Set up callbacks
    # callbacks = [
    #     EarlyStoppingCallback(
    #         early_stopping_patience=3,
    #         early_stopping_threshold=0.01
    #     ),
    #     # Select a few examples for tracking inference during training
    #     InferenceCallback(
    #         validation_examples=test_dataset.select(range(5)),
    #         tokenizer=tokenizer,
    #         log_steps=500
    #     )
    # ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # ---------- ADD EVALUATION METRICS AND CALLBACKS ----------
        # compute_metrics=compute_metrics,  # Add the evaluation metrics
        # callbacks=callbacks,  # Add callbacks for early stopping and inference
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")
    
    
    # ---------- DPO TRAINING (OPTIONAL) ----------
    # If we  want to implement DPO after SFT:
    # print("Starting DPO training...")
    # 
    # # Prepare DPO datasets
    # dpo_train_dataset, dpo_eval_dataset = prepare_dpo_data(train_df)
    # 
    # # Initialize DPO trainer
    # ...
    # 
    # # Train with DPO
    # ...
    # 
    # # Save the DPO-trained model
    # ...
    # print(f"DPO training complete! Model saved to {OUTPUT_DIR}-dpo")
    
    return model, trainer

if __name__ == "__main__":
    train()
