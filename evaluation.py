import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from few_shot_system import HybridTopicClassifier
from prompting import PromptConstructor, ExampleDatabase
from self_consistency import SelfConsistencyFramework

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "output/llama-3.2-3b-alpaca-lora"
MAX_NEW_TOKENS = 200
NUM_SAMPLES = 3  # For self-consistency
DEFAULT_EXAMPLES = 3
USE_SELF_CONSISTENCY = True  # To use self-consistency

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for test data")
    parser.add_argument("--input", type=str, default="test_data_student.json", 
                        help="Path to test data JSON file")
    parser.add_argument("--output", type=str, default="predictions.json", 
                        help="Path to save predictions")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES, 
                        help="Number of examples to use")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, 
                        help="Number of samples for self-consistency")
    parser.add_argument("--consistency", action="store_true", default=USE_SELF_CONSISTENCY, 
                        help="Enable self-consistency")
    parser.add_argument("--cot", action="store_true", default=True, 
                        help="Enable chain-of-thought reasoning")
    return parser.parse_args()

def load_model_components():
    """Load and initialize all model components"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Initializing system components...")
    classifier = HybridTopicClassifier()
    db = ExampleDatabase("example_database.json")
    pc = PromptConstructor(db, classifier)
    
    # Initialize self-consistency framework if needed
    sc_framework = None
    if USE_SELF_CONSISTENCY:
        sc_framework = SelfConsistencyFramework(model, tokenizer, pc)
    
    return model, tokenizer, classifier, pc, sc_framework

def generate_basic_response(model, tokenizer, pc, query, topic=None, num_examples=3, use_cot=True):
    """Generate a response using basic prompting"""
    # Construct prompt
    prompt = pc.construct_prompt(
        query, 
        topic=topic,
        num_examples=num_examples,
        use_cot=use_cot
    )
    
    # Run inference
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_response = full_response.replace(prompt, "").strip()
    
    # Extract final answer if using CoT
    if use_cot:
        final_answer = pc.extract_final_answer(model_response)
        return final_answer
    
    return model_response

def process_test_data(args):
    """Process the test data and generate predictions"""
    print(f"Loading test data from {args.input}...")
    with open(args.input, "r") as f:
        test_data = json.load(f)
    
    print(f"Loading model and components...")
    model, tokenizer, classifier, pc, sc_framework = load_model_components()
    
    print(f"Generating predictions for {len(test_data)} questions...")
    predictions = []
    
    for item in tqdm(test_data, desc="Processing"):
        question = item["question"]
        
        # Classify topic
        topic, _ = classifier.classify(question)
        
        # Choose generation strategy based on configuration
        if args.consistency and sc_framework:
            # Generate with self-consistency
            results = sc_framework.generate_with_self_consistency(
                question, 
                topic=topic, 
                num_samples=args.samples
            )
            prediction = results["final_answer"]
        else:
            # Generate with basic prompting
            prediction = generate_basic_response(
                model, 
                tokenizer, 
                pc, 
                question, 
                topic=topic,
                num_examples=args.examples,
                use_cot=args.cot
            )
        
        # Clean up prediction (remove any instruction artifacts)
        prediction = clean_prediction(prediction)
        
        # Add to results
        predictions.append({"prediction": prediction})
    
    # Save predictions
    print(f"Saving predictions to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    
    # Validate format
    validate_output(args.input, args.output)
    
    print("Evaluation completed successfully!")
    return predictions

def clean_prediction(prediction):
    """Clean up the prediction string to ensure it's properly formatted"""
    # Remove any trailing or leading whitespace
    prediction = prediction.strip()
    
    # Remove any "The answer is" or similar prefixes
    prefixes = ["The answer is", "Answer:", "Therefore,", "Thus,", "Hence,", "Finally,"]
    for prefix in prefixes:
        if prediction.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
    
    # Remove special tokens or other artifacts
    prediction = prediction.replace("<s>", "").replace("</s>", "")
    
    # For multiple choice, ensure just the letter is provided if that's the pattern
    if re.match(r'^[A-D][\.\s]', prediction):
        prediction = prediction[0]
    
    return prediction

def validate_output(input_path, output_path):
    """Validate that the output format meets the requirements"""
    try:
        with open(input_path, "r") as f:
            source = json.load(f)
        with open(output_path, "r") as f:
            prediction = json.load(f)
        
        assert len(source) == len(prediction)
        for pred in prediction:
            assert "prediction" in pred
            assert type(pred["prediction"]) == str
        print("Output validation passed")
    except Exception as e:
        print(f"Output validation failed: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    process_test_data(args)
