import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting import PromptConstructor, ExampleDatabase, HybridTopicClassifier
from rouge_score import rouge_scorer

MODEL_NAME = "meta-llama/Llama-3.2-3B"
MAX_NEW_TOKENS = 200


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# get examples
example_cache = {}

def get_cached_examples(pc: PromptConstructor, query: str, topic: str, num_examples: int):
    # Check if we have cached examples for this query type
    
    query_type = (topic, query, num_examples)
    if query_type in example_cache:
        return example_cache[query_type]

    # If not, select examples and cache them
    examples = pc.example_db.select_relevant_examples(query, topic, n=num_examples)
    example_cache[query_type] = examples
    return examples

# prompt generation
def generate_response(pc: PromptConstructor, query: str, num_examples: int = 3):
    topic, _ = pc.classifier.classify(query)
    examples = get_cached_examples(pc, query, topic, num_examples)
    
    # force-inject selected examples into the prompt constructor
    prompt = pc.construct_prompt(query, topic)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # strip the prompt from the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only = response.replace(prompt, "").strip()
    return generated_only

# exact match metric
def dummy_metric_fn(preds, refs):
    return {
        "exact_match": sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs)
    }
# rouge L metric
def rouge_l_metric(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(preds, refs)]
    return {
        "rougeL_f1_avg": sum(scores) / len(scores)
    }
# eval, change metric
def evaluate_few_shot_prompting(test_queries, configs, pc: PromptConstructor, metric_fn=rouge_l_metric):
    results = {}

    for config in configs:
        # Configure the system with different parameters
        # Run inference on test queries
        # Compute metrics
        num_examples = config.get("num_examples", 3)

        predictions = []
        references = []

        for query, expected in test_queries:
            prediction = generate_response(pc, query, num_examples=num_examples)
            predictions.append(prediction)
            references.append(expected)
            print("Prompt:\n", query)
            print("Model Output:\n", prediction)
            print("Reference:\n", expected)

        metrics = metric_fn(predictions, references)
        results[str(config)] = metrics

    return results

# main testing
if __name__ == "__main__":
    classifier = HybridTopicClassifier()
    db = ExampleDatabase()
    pc = PromptConstructor(db, classifier)

    test_queries = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Translate to Spanish: Hello", "Hola")
    ]


    # parameters for testing
    # {1-5, random/similarity/diversity, examples_first/query_first}
    configs = [
        {"num_examples": 2, "strategy": "random", "prompt_format": "examples_first"},
        {"num_examples": 4, "strategy": "similarity", "prompt_format": "query_first"}
    ]

    results = evaluate_few_shot_prompting(test_queries, configs, pc)

    for cfg, metrics in results.items():
        print(f"{cfg} => {metrics}")