import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

from few_shot_system import HybridTopicClassifier

# Template for dynamic prompt construction
# [TOPIC], [EXAMPLES], and [QUERY] are dynamically replaced with real content
PROMPT_TEMPLATE_EXAMPLES_FIRST = """Below are examples of [TOPIC] tasks:

[EXAMPLES]

Now perform this new task:
[QUERY]
"""
PROMPT_TEMPLATE_QUERY_FIRST = """You are given a task below:

[QUERY]

Here are similar examples:

[EXAMPLES]
"""

MODEL_NAME = "meta-llama/Llama-3.2-3B"
MAX_LENGTH = 2048  # Adjust this based on the actual model context window

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception as e:
    print("Tokenizer load failed. Ensure Hugging Face auth or local files are set.")
    raise e

class ExampleDatabase:
    """
    Loads and manages example data used for few-shot prompting.
    Supports retrieval and similarity-based selection by topic.
    """
    def __init__(self, json_path="example_database.json"):
        with open(json_path, "r") as f:
            self.examples = json.load(f)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
        self.topic_index = self._index_by_topic()

    def _index_by_topic(self):
        # Organize examples by topic into a dictionary
        topic_index = {}
        for ex in self.examples:
            topic = ex["topic"]
            topic_index.setdefault(topic, []).append(ex)
        return topic_index

    def retrieve_examples(self, topic):
        # Return all examples for a given topic
        return self.topic_index.get(topic, [])

    # example selection for similarity (default), random, or diversity
    def select_relevant_examples(self, query, topic, n=3, strategy = "similarity"):
        """
        Select top-n most semantically similar examples for a query within a topic.
        """
        all_examples = self.retrieve_examples(topic)
        if not all_examples:
            return []
        # support for random and diversity strategies
        if strategy == "random":
            return np.random.choice(all_examples, size=min(n, len(all_examples)), replace=False).tolist()

        if strategy == "diversity":
            embeddings = self.model.encode([ex["instruction"] for ex in all_examples])
            diverse_indices = [0]
            while len(diverse_indices) < min(n, len(all_examples)):
                remaining = [i for i in range(len(all_examples)) if i not in diverse_indices]
                dists = [min(np.linalg.norm(embeddings[i] - embeddings[j]) for j in diverse_indices) for i in remaining]
                diverse_indices.append(remaining[np.argmax(dists)])
            return [all_examples[i] for i in diverse_indices]
        
        # OG similarity
        # Embed the query and example instructions
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        example_embeddings = self.model.encode(
            [ex["instruction"] for ex in all_examples], convert_to_tensor=True
        )

        # Compute cosine similarity and return top-n examples
        similarities = util.cos_sim(query_embedding, example_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:n]
        return [all_examples[i] for i in top_indices]
    
# constructs prompt based on chosen formats example first, query first, with/without explanation
class PromptConstructor:
    """
    Handles prompt assembly using topic classification and example selection.
    """
    def __init__(self, example_db: ExampleDatabase, classifier: HybridTopicClassifier):
        self.example_db = example_db
        self.classifier = classifier

    def format_examples(self, examples, with_explanations=False):
        # Format examples for insertion into prompt template
        formatted = []
        for ex in examples:
            explanation = f"\nExplanation: {ex['explanation']}" if with_explanations and 'explanation' in ex else ""
            formatted.append(f"Instruction: {ex['instruction']}\nResponse: {ex['response']}{explanation}")
        return "\n\n".join(formatted)

    def format_cot_examples(self, examples):
        # Format examples with step-by-step reasoning for CoT prompting
        formatted = []
        for ex in examples:
            if 'explanation' in ex:
                formatted.append(f"Instruction: {ex['instruction']}\nReasoning: {ex['explanation']}\nResponse: {ex['response']}")
            else:
                # Fall back to non-CoT format if explanation is missing
                formatted.append(f"Instruction: {ex['instruction']}\nResponse: {ex['response']}")
        return "\n\n".join(formatted)

    def construct_prompt(self, query, topic=None, strategy="similarity", num_examples=3, 
                         prompt_format="examples_first", with_explanations=False, use_cot=False):
        """
        Build a complete prompt with few-shot examples and the new query.
        Truncates examples if necessary to fit the model context window.
        
        Parameters:
        - query: The user's query
        - topic: Topic classification (if None, will be automatically classified)
        - strategy: How to select examples ("similarity", "random", "diversity")
        - num_examples: Number of few-shot examples to include
        - prompt_format: Format of the prompt ("examples_first" or "query_first")
        - with_explanations: Whether to include explanations in examples
        - use_cot: Whether to use Chain-of-Thought prompting
        """
        # Classify topic if not already provided
        if topic is None:
            topic, _ = self.classifier.classify(query)

        # Retrieve examples
        examples = self.example_db.select_relevant_examples(query, topic, n=num_examples, strategy=strategy)
        
        # Choose formatting based on whether we're using CoT
        if use_cot:
            examples_text = self.format_cot_examples(examples)
            prompt = COT_PROMPT_TEMPLATE.replace("[TOPIC]", topic)
            prompt = prompt.replace("[EXAMPLES]", examples_text)
            prompt = prompt.replace("[QUERY]", query)
        else:
            examples_text = self.format_examples(examples, with_explanations=with_explanations)
            
            if prompt_format == "query_first":
                prompt = PROMPT_TEMPLATE_QUERY_FIRST.replace("[TOPIC]", topic)
                prompt = prompt.replace("[QUERY]", query)
                prompt = prompt.replace("[EXAMPLES]", examples_text)
            else:
                prompt = PROMPT_TEMPLATE_EXAMPLES_FIRST.replace("[TOPIC]", topic)
                prompt = prompt.replace("[EXAMPLES]", examples_text)
                prompt = prompt.replace("[QUERY]", query)

        # Ensure prompt fits within model context length
        while len(tokenizer.encode(prompt)) > MAX_LENGTH and len(examples) > 0:
            examples.pop()  # Remove least relevant
            
            if use_cot:
                examples_text = self.format_cot_examples(examples)
                prompt = COT_PROMPT_TEMPLATE.replace("[TOPIC]", topic)
                prompt = prompt.replace("[EXAMPLES]", examples_text)
                prompt = prompt.replace("[QUERY]", query)
            else:
                examples_text = self.format_examples(examples, with_explanations=with_explanations)
                
                if prompt_format == "query_first":
                    prompt = PROMPT_TEMPLATE_QUERY_FIRST.replace("[TOPIC]", topic)
                    prompt = prompt.replace("[QUERY]", query)
                    prompt = prompt.replace("[EXAMPLES]", examples_text)
                else:
                    prompt = PROMPT_TEMPLATE_EXAMPLES_FIRST.replace("[TOPIC]", topic)
                    prompt = prompt.replace("[EXAMPLES]", examples_text)
                    prompt = prompt.replace("[QUERY]", query)

        return prompt

    def extract_final_answer(self, response):
    """
    Extract the final answer from Chain-of-Thought response.
    Uses multiple strategies to identify the conclusion.
    
    Args:
        response (str): The model's response including reasoning steps
        
    Returns:
        str: The extracted final answer
    """
    if not response or response.isspace():
        return "No answer provided"
    
    # Look for conclusion indicators
    indicators = [
        "Therefore,", "Thus,", "So,", "Hence,", "In conclusion,", 
        "The answer is", "The final answer is", "To conclude,", "Finally,",
        "This means", "We can conclude", "The result is", "The solution is"
    ]
    
    # Check for the last occurrence of any indicator
    last_indicator_pos = -1
    last_indicator = None
    
    lines = response.split('\n')
    for i, line in enumerate(lines):
        for indicator in indicators:
            if indicator.lower() in line.lower():
                if i > last_indicator_pos:
                    last_indicator_pos = i
                    last_indicator = indicator
    
    # If we found an indicator, extract text after it
    if last_indicator_pos >= 0:
        line = lines[last_indicator_pos]
        # Extract everything after the indicator
        indicator_pos = line.lower().find(last_indicator.lower())
        if indicator_pos >= 0:
            extracted = line[indicator_pos + len(last_indicator):].strip()
            if extracted:  # If there's content after the indicator
                return extracted
    
    # Check for lines starting with "Answer:"
    for line in reversed(lines):  # Start from the end
        if line.lower().startswith("answer:"):
            return line[7:].strip()
    
    # Check if the last paragraph is short (likely a conclusion)
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1].strip()
        # If the last line is short, it's likely a conclusion
        if len(last_line) < 100 and not last_line.startswith("Let's") and not last_line.startswith("Step"):
            return last_line
    
    # Look for the last sentence in the response
    if response:
        sentences = response.split('.')
        if sentences:
            # Get the last non-empty sentence
            for sentence in reversed(sentences):
                if sentence.strip():
                    return sentence.strip()
    
    # Return last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    
    # Ultimate fallback
    return response.strip()

# For testing and demonstration only, will not run during imports
if __name__ == "__main__":
    classifier = HybridTopicClassifier()
    db = ExampleDatabase()
    pc = PromptConstructor(db, classifier)

    test_query = "Translate this sentence into Spanish: The sun is shining."
    constructed = pc.construct_prompt(test_query)
    print("\nConstructed Prompt:\n")
    print(constructed)
