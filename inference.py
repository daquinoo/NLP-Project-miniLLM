import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting import PromptConstructor, ExampleDatabase, HybridTopicClassifier
from peft import PeftModel
import argparse

MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "output/llama-3.2-3b-alpaca-lora"
MAX_NEW_TOKENS = 300

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot prompting with optional chain-of-thought reasoning")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought reasoning")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples to use (default: 3)")
    parser.add_argument("--extract_answer", action="store_true", help="Extract final answer from CoT response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--format", choices=["examples_first", "query_first"], default="examples_first", 
                        help="Prompt format (default: examples_first)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Checking GPU availability...")
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    
    # Load base model and LoRA adapter
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Base model loaded on device:", next(base_model.parameters()).device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("LoRA adapter loaded on device:", next(model.parameters()).device)
    model.eval()
    
    # Initialize pipeline components
    classifier = HybridTopicClassifier()
    db = ExampleDatabase()
    pc = PromptConstructor(db, classifier)
    
    cot_status = "ENABLED" if args.cot else "DISABLED"
    extract_status = "ENABLED" if args.extract_answer else "DISABLED"
    
    print(f"\nWelcome to the Few-Shot Prompting System (LoRA-tuned) 🚀")
    print(f"Chain-of-Thought: {cot_status} | Examples: {args.examples} | Answer Extraction: {extract_status}")
    print(f"Temperature: {args.temperature} | Format: {args.format}\n")
    
    while True:
        try:
            query = input("Enter your query (or type 'exit'): ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            
            # Classify topic
            topic, confidence = classifier.classify(query)
            print(f"\nDetected topic: {topic} (confidence: {confidence:.2f})")
            
            # Construct prompt with or without CoT
            prompt = pc.construct_prompt(
                query, 
                topic=topic,
                num_examples=args.examples,
                prompt_format=args.format,
                use_cot=args.cot
            )
            
            print("\nConstructed Prompt:\n")
            print(prompt)
            
            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    top_p=0.9
                )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_response = full_response.replace(prompt, "").strip()
            
            print("\nModel Response:\n")
            print(model_response)
            
            # Extract final answer if CoT is enabled and extraction is requested
            if args.cot and args.extract_answer:
                final_answer = pc.extract_final_answer(model_response)
                print("\nExtracted Final Answer:\n")
                print(final_answer)
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
