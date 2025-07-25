import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting import PromptConstructor, ExampleDatabase, HybridTopicClassifier
from peft import PeftModel
import argparse
from self_consistency import SelfConsistencyFramework
from verification import AnswerVerifier
from safety import SafetyFilter

MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "output/llama-3.2-3b-alpaca-lora"
MAX_NEW_TOKENS = 50

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot prompting with optional chain-of-thought reasoning")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought reasoning")
    parser.add_argument("--consistency", action="store_true", help="Enable self-consistency with multiple paths")
    parser.add_argument("--verify", action="store_true", help="Enable answer verification")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples for self-consistency (default: 5)")
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

    # Initialize safety filter
    safety_filter = SafetyFilter("safety_database.json")
    
    # Initialize pipeline components
    classifier = HybridTopicClassifier()
    db = ExampleDatabase()
    pc = PromptConstructor(db, classifier)
    
    # Initialize enhanced components if needed
    if args.consistency:
        sc_framework = SelfConsistencyFramework(model, tokenizer, pc)
    if args.verify:
        verifier = AnswerVerifier()
    
    # Print configuration
    cot_status = "ENABLED" if args.cot else "DISABLED"
    consistency_status = "ENABLED" if args.consistency else "DISABLED"
    verify_status = "ENABLED" if args.verify else "DISABLED"
    extract_status = "ENABLED" if args.extract_answer else "DISABLED"
    
    print(f"\nWelcome to the Few-Shot Prompting System (LoRA-tuned) 🚀")
    print(f"Chain-of-Thought: {cot_status} | Self-Consistency: {consistency_status} | Verification: {verify_status}")
    print(f"Examples: {args.examples} | Answer Extraction: {extract_status}")
    print(f"Temperature: {args.temperature} | Format: {args.format}")
    if args.consistency:
        print(f"Samples for Self-Consistency: {args.samples}")
    print("\n")
    
    while True:
        try:
            query = input("Enter your query (or type 'exit'): ").strip()
            
            # Catch empty input
            if not query:
                print("\n⚠️ Empty query received. Please type a valid question.")
                print("\n" + "="*60 + "\n")
                continue
            
            if query.lower() in ["exit", "quit"]:
                break

            # Safety check for query
            safe, refusal_response = safety_filter.check_query(query)
            if not safe:
                print(f"\n⚠️ {refusal_response}")
                with open("refusal_log.txt", "a") as f:
                    f.write(f"Unsafe Query: {query}\n")
                    f.write(f"Refusal Message: {refusal_response}\n")
                    f.write("="*50 + "\n")
                print("\n" + "="*60 + "\n")
                continue
            
            # Classify topic
            topic, confidence = classifier.classify(query)
            print(f"\nDetected topic: {topic} (confidence: {confidence:.2f})")
            
            # Different processing paths based on configuration
            if args.consistency:
                # Full self-consistency pipeline
                print(f"\nGenerating {args.samples} reasoning paths...")
                results = sc_framework.generate_with_self_consistency(
                    query, 
                    topic=topic, 
                    num_samples=args.samples
                )
                
                print("\n=== Self-Consistency Results ===")
                print(f"Final Answer: {results['final_answer']}")
                print(f"Confidence: {results['confidence_level']} ({results['confidence_score']:.2f})")
                
                if args.verify:
                    # Verify the answer against reasoning
                    # Use the highest-confidence reasoning path
                    best_reasoning = results['reasoning_paths'][0]
                    verification = verifier.verify_answer(best_reasoning, results['final_answer'])
                    
                    print("\n=== Verification Results ===")
                    print(f"Verification: {verification['verified']}")
                    print(f"Reason: {verification['reason']}")
                
                # Option to show all reasoning paths
                show_all = input("\nShow all reasoning paths? (y/n): ").lower() == 'y'
                if show_all:
                    for i, (path, answer) in enumerate(zip(results['reasoning_paths'], results['extracted_answers'])):
                        print(f"\n--- Reasoning Path {i+1} ---")
                        print(path)
                        print(f"\nExtracted Answer: {answer}")
                        print("-" * 40)
            
            else:
                # Standard CoT or few-shot
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

                # Safety check for model response
                safe_response, refusal_response = safety_filter.check_query(model_response)
                if not safe_response:
                    print(f"\n⚠️ {refusal_response}")
                    with open("refusal_log.txt", "a") as f:
                        f.write(f"Unsafe Model Output: {model_response}\n")
                        f.write(f"Refusal Message: {refusal_response}\n")
                        f.write("="*50 + "\n")
                    print("\n" + "="*60 + "\n")
                    continue
                
                # Handle answer extraction and verification
                if args.cot:
                    if args.extract_answer:
                        final_answer = pc.extract_final_answer(model_response)
                        print("\n=== Extracted Answer ===")
                        print(final_answer)
                        
                        if args.verify:
                            verification = verifier.verify_answer(model_response, final_answer)
                            print("\n=== Verification Results ===")
                            print(f"Verification: {verification['verified']}")
                            print(f"Reason: {verification['reason']}")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
