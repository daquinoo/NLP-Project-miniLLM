import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import Dataset, load_dataset
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluate import load
import evaluate
from transformers import EarlyStoppingCallback, TrainerCallback
from peft import PeftModel
#
import nltk
nltk.download('punkt')
# from trl import DPOTrainer  # For Direct Preference Optimization

ds = load_dataset("tatsu-lab/alpaca")
# remove duplicates
text_df = pd.DataFrame(ds)
text_df.head(5)
text_df = text_df['train'].apply(pd.Series)
print("Size of original dataset: " + str(text_df.size))
text_df = text_df.drop("text", axis = 1)
# remove lines where input OR output is empty
text_df = text_df.replace('', None).dropna(subset=['instruction', 'output'])
text_df = text_df.drop_duplicates()
print("Size of dataset after removing duplicates and empty input/output: " + str(text_df.size))
text_df.head(5)
# format to input - output pairst
# include instruction, context, response
# input = instruction column
# context = input column
# response = output column
text_df = text_df.rename(columns={'instruction': 'instruction', 'input': 'context', 'output': 'response'})
text_df.head(5)
# 80 - 20 random validation split
train_df, test_df = train_test_split(text_df, test_size=0.2, random_state=52)
print("Size of train dataset: " + str(train_df.size))
print("Size of test dataset: " + str(test_df.size))
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
print("Saved preprocessed data to CSV files")
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
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
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
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
def compute_metrics(eval_preds):
    """
    Compute ROUGE, BLEU, BERTScore, and (optionally) Perplexity.
    """
    predictions, labels = eval_preds

    # Convert tokens to text if needed
    if isinstance(predictions[0], list):  # If tokenized
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    if isinstance(labels[0], list):
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=labels)

    # BLEU (simplified using sentence BLEU)
    bleu_scores = [
        sentence_bleu([word_tokenize(label)], word_tokenize(pred))
        for pred, label in zip(predictions, labels)
    ]
    avg_bleu = np.mean(bleu_scores)

    # BERTScore
    bertscore = evaluate.load("bertscore")
    bertscore_result = bertscore.compute(predictions=predictions, references=labels, lang="en")
    
    # perplexity

    metrics = {
        "bleu": avg_bleu,
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bertscore_f1": np.mean(bertscore_result["f1"]),
        # Optional: add perplexity if tracking via eval loss externally
    }

    return metrics

if __name__ == "__main__":
    # TEST ONLY: Evaluate metrics function without loading full train_df
    from types import SimpleNamespace

    dummy_preds = ["The weather is nice today.", "The cat is on the mat."]
    dummy_labels = ["It's a nice day outside.", "The cat sat on the mat."]

    class DummyEvalPreds(SimpleNamespace):
        def __iter__(self):
            return iter((self.predictions, self.label_ids))

    dummy = DummyEvalPreds(predictions=dummy_preds, label_ids=dummy_labels)

    metrics = compute_metrics(dummy)
    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


# ---------- INFERENCE CALLBACK FOR VALIDATION EXAMPLES ----------
# Implement a callback for periodic inference on held-out 
# NP-4/7/2025
class InferenceCallback(TrainerCallback):
    """
        Args:
            tokenizer: The tokenizer used to prepare input for the model.
            val_texts: A list of sample input prompts for qualitative evaluation.
            log_steps: How often (in steps) to run inference and print results.
    """
    def __init__(self, tokenizer, val_texts, log_steps=500):
        self.tokenizer = tokenizer
        self.val_texts = val_texts  # List of sample validation prompts
        self.log_steps = log_steps
        
    
    """
        Called at the end of each training step. If the step number matches log_steps,
        this runs inference on the sample prompts and prints model outputs.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            model = kwargs["model"]

            # Ensure compatibility with available device
            device = model.device if torch.cuda.is_available() else "cpu"
            model.eval()

            print(f"\n--- Inference at step {state.global_step} ---")
            with torch.no_grad(): # Disable gradient computation
                for i, prompt in enumerate(self.val_texts):
                    # Tokenize and move input to correct device
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

                    # Generate output tokens
                    outputs = model.generate(**inputs, max_new_tokens=50)

                    # Decode and print the generated text
                    decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Prompt {i+1}: {prompt}")
                    print(f"Output {i+1}: {decoded}")
                    print("---")


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
        learning_rate=2e-5,           # Learning rate - typically higher for LoRA fine-tuning, change from default (2e-4) to whichever is preferred (1e-5 to 5e-5, or otherwise)
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
        # addressing above comments - threshold is not supported by TrainingArguments, created a method for patience (get_callbacks) which 
        # will be passed in the trainer - NP-4/7/2025
    )
    
# Define callbacks including early stopping NP- 4/7/2025
# Returns a list of Trainer callbacks including:
# - EarlyStoppingCallback: stops training if no improvement in eval loss
# - InferenceCallback (optional): logs model outputs on sample prompts during training
def get_callbacks(tokenizer=None, val_texts=None):
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] # stop if no improvelemt for 3 evals
    
    # If sample prompts and tokenizer are provided, add inference logging
    if tokenizer and val_texts:
        callbacks.append(InferenceCallback(tokenizer, val_texts, log_steps=500))
    return callbacks


# Main training function
def train():
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(train_df, test_df)
    
    # Prepare model
    model = prepare_model()
    
    # Get training arguments
    training_args = get_training_args()
    
    # ---------- SAMPLE PROMPTS FOR INFERENCE CALLBACK ----------
    # These are fixed input prompts used by InferenceCallback to monitor the model's
    # qualitative performance during training. They cover a variety of task types.
    val_examples = [
        "Translate to French: The book is on the table.",
        "Summarize: Alice went to the store and bought apples, oranges, and bananas.",
        "Write a poem about the moon.",
        "Explain what a black hole is.",
        "Convert the temperature from Celsius to Fahrenheit: 20°C."
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # ---------- ADD EVALUATION METRICS AND CALLBACKS ----------
        compute_metrics=compute_metrics,  # Add the evaluation metrics
        callbacks=get_callbacks(tokenizer, val_examples),  # Add callbacks for early stopping and inference
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
