import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting import PromptConstructor, ExampleDatabase, HybridTopicClassifier
from peft import PeftModel

MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "output/llama-3.2-3b-alpaca-lora"
MAX_NEW_TOKENS = 300

def main():
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

    print("\nWelcome to the Few-Shot Prompting System (LoRA-tuned) 🚀\n")
    while True:
        try:
            query = input("Enter your query (or type 'exit'): ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            topic, confidence = classifier.classify(query)
            print(f"\nDetected topic: {topic} (confidence: {confidence:.2f})")

            prompt = pc.construct_prompt(query, topic=topic)
            print("\nConstructed Prompt:\n")
            print(prompt)

            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_response = response.replace(prompt, "").strip()

            print("\nModel Response:\n")
            print(final_response)
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
