import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

from few_shot_system import HybridTopicClassifier

# Template for dynamic prompt construction
# [TOPIC], [EXAMPLES], and [QUERY] are dynamically replaced with real content
PROMPT_TEMPLATE = """Below are examples of [TOPIC] tasks:

[EXAMPLES]

Now perform this new task:
[QUERY]
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

    def select_relevant_examples(self, query, topic, n=3):
        """
        Select top-n most semantically similar examples for a query within a topic.
        """
        all_examples = self.retrieve_examples(topic)
        if not all_examples:
            return []

        # Embed the query and example instructions
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        example_embeddings = self.model.encode(
            [ex["instruction"] for ex in all_examples], convert_to_tensor=True
        )

        # Compute cosine similarity and return top-n examples
        similarities = util.cos_sim(query_embedding, example_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:n]
        return [all_examples[i] for i in top_indices]

class PromptConstructor:
    """
    Handles prompt assembly using topic classification and example selection.
    """
    def __init__(self, example_db: ExampleDatabase, classifier: HybridTopicClassifier):
        self.example_db = example_db
        self.classifier = classifier

    def format_examples(self, examples):
        # Format examples for insertion into prompt template
        return "\n\n".join([
            f"Instruction: {ex['instruction']}\nResponse: {ex['response']}"
            for ex in examples
        ])

    def construct_prompt(self, query, topic=None):
        """
        Build a complete prompt with few-shot examples and the new query.
        Truncates examples if necessary to fit the model context window.
        """
        # Classify topic if not already provided
        if topic is None:
            topic, _ = self.classifier.classify(query)

        # Retrieve and format relevant examples
        examples = self.example_db.select_relevant_examples(query, topic)
        examples_text = self.format_examples(examples)

        # Construct the prompt using the template
        prompt = PROMPT_TEMPLATE.replace("[TOPIC]", topic)
        prompt = prompt.replace("[EXAMPLES]", examples_text)
        prompt = prompt.replace("[QUERY]", query)

        # Ensure prompt fits within model context length
        while len(tokenizer.encode(prompt)) > MAX_LENGTH and len(examples) > 0:
            examples.pop()  # Remove least relevant
            examples_text = self.format_examples(examples)
            prompt = PROMPT_TEMPLATE.replace("[TOPIC]", topic)
            prompt = prompt.replace("[EXAMPLES]", examples_text)
            prompt = prompt.replace("[QUERY]", query)

        return prompt

# For testing and demonstration only, will not run during imports
if __name__ == "__main__":
    classifier = HybridTopicClassifier()
    db = ExampleDatabase()
    pc = PromptConstructor(db, classifier)

    test_query = "Translate this sentence into Spanish: The sun is shining."
    constructed = pc.construct_prompt(test_query)
    print("\nConstructed Prompt:\n")
    print(constructed)
