import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Import from your own module
from few_shot_system import HybridTopicClassifier

# Template for dynamic prompt construction
# [TOPIC], [EXAMPLES], and [QUERY] are dynamically replaced with real content
PROMPT_TEMPLATE_EXAMPLES_FIRST = """Below are examples of [TOPIC] tasks:

[EXAMPLES]

Here is your task to answer directly:
[QUERY]
"""
PROMPT_TEMPLATE_QUERY_FIRST = """You are given a task below:

[QUERY]

Here are similar examples:

[EXAMPLES]
"""
COT_PROMPT_TEMPLATE = """Below are examples of [TOPIC] tasks with step-by-step reasoning:

[EXAMPLES]

Your task is to answer the following question:
[QUERY]
Let's think through this step by step to reach the correct answer.
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
    Supports retrieval and TF-IDF similarity-based selection by topic.
    """
    def __init__(self, json_path="example_database.json"):
        with open(json_path, "r") as f:
            self.examples = json.load(f)
        
        # Organize examples by topic
        self.topic_index = self._index_by_topic()
        
        # Initialize TF-IDF vectorizers per topic (to avoid cross-topic comparison issues)
        self.vectorizers = {}
        self.example_vectors = {}
        
        # Precompute TF-IDF vectors for each topic
        for topic, examples in self.topic_index.items():
            if examples:
                # Create vectorizer with bigrams for better semantic matching
                self.vectorizers[topic] = TfidfVectorizer(ngram_range=(1, 2))
                # Get instructions text for each example
                instructions = [ex["instruction"] for ex in examples]
                # Compute and store vectors
                self.example_vectors[topic] = self.vectorizers[topic].fit_transform(instructions)

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

    def select_relevant_examples(self, query, topic, n=3, strategy="similarity"):
        """
        Select examples for a query within a topic based on strategy.
        Strategies: "similarity" (TF-IDF), "random", or "diversity"
        """
        all_examples = self.retrieve_examples(topic)
        if not all_examples:
            return []
        
        # Random selection strategy
        if strategy == "random":
            return np.random.choice(all_examples, size=min(n, len(all_examples)), replace=False).tolist()
        
        # Diversity strategy (using TF-IDF vectors)
        if strategy == "diversity":
            if topic not in self.vectorizers:
                return np.random.choice(all_examples, size=min(n, len(all_examples)), replace=False).tolist()
            
            # Start with a random example
            selected_indices = [np.random.randint(0, len(all_examples))]
            remaining_indices = list(set(range(len(all_examples))) - set(selected_indices))
            
            # Select examples maximizing diversity
            while len(selected_indices) < min(n, len(all_examples)) and remaining_indices:
                # Get vectors for selected examples
                selected_vectors = self.example_vectors[topic][selected_indices]
                
                # Find example with maximum distance from already selected
                max_min_dist = -1
                next_idx = -1
                
                for i in remaining_indices:
                    # Get vector for candidate example
                    candidate_vector = self.example_vectors[topic][i]
                    
                    # Compute similarities with already selected examples
                    sims = cosine_similarity(candidate_vector, selected_vectors)[0]
                    
                    # Get minimum similarity (maximum diversity)
                    min_sim = min(sims)
                    
                    # Transform similarity to distance (1 - similarity)
                    min_dist = 1 - min_sim
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        next_idx = i
                
                if next_idx != -1:
                    selected_indices.append(next_idx)
                    remaining_indices.remove(next_idx)
                else:
                    break
            
            return [all_examples[i] for i in selected_indices]
        
        # Default: similarity-based selection using TF-IDF
        if topic not in self.vectorizers:
            return np.random.choice(all_examples, size=min(n, len(all_examples)), replace=False).tolist()
        
        # Transform query using the topic's vectorizer
        query_vector = self.vectorizers[topic].transform([query])
        
        # Compute similarities with all examples in the topic
        similarities = cosine_similarity(query_vector, self.example_vectors[topic])[0]
        
        # Get top-n examples
        top_indices = similarities.argsort()[-n:][::-1]
        return [all_examples[i] for i in top_indices]

# PromptConstructor class (unchanged)
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

    def extract_final_answer(self, response, query=None):
        """
        Extract the final answer from Chain-of-Thought response.
        Uses multiple strategies to identify the conclusion.
        
        Args:
            response (str): The model's response
            query (str): Optional original query to help identify relevant section
        """
        if not response or response.isspace():
        return "No answer provided"
    
        # Apply stopper if needed (handle model drift)
        stop_phrases = ["Now perform", "Next task", "Next question", "Instruction:"]
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        # SPECIAL HANDLING FOR MULTIPLE CHOICE QUESTIONS
        # Look for option selection patterns
        option_patterns = [
            r'(?:answer|option|choice)(?:\s+is)?(?:\s*:)?\s*([A-D])',
            r'(?:select|choose|pick)(?:\s+option)?(?:\s*:)?\s*([A-D])',
            r'(?:correct|right)(?:\s+answer|option|choice)?(?:\s+is)?(?:\s*:)?\s*([A-D])',
            r'(?:the)?\s*([A-D])(?:\s+is)(?:\s+the)(?:\s+correct|right|answer)',
            r'(?:^|\s+|\.)([A-D])(?:\s*\.|\s*$|\s*,)',
            r'→\s*([A-D])',
            r'option\s*([A-D])',
            r'\b([A-D])[\.:]'
        ]
        
        for pattern in option_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # Return just the letter for multiple choice
        
        # For numerical answers, look for specific patterns
        numeric_patterns = [
            r'(?:answer|result|value)(?:\s+is)?(?:\s*:)?\s*(\d+)',
            r'(?:=\s*)(\d+)(?:\s*\.|\s*$)',
            r'(?:^|\s+|\.)(\d+)(?:\s*is\s*the\s*answer)',
            r'(?:final\s*answer)(?:\s+is)?(?:\s*:)?\s*(\d+)'
        ]
        
        for pattern in numeric_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # First, try to find and isolate the section relevant to our query
        if query:
            query_lower = query.lower()
            # Split by potential separators that might indicate new examples
            sections = re.split(r'Now perform (?:this|the next) task:|Instruction:', response)
            
            if len(sections) > 1:
                # Find the most relevant section
                best_section = sections[0]  # Default to first section
                best_score = 0
                
                for section in sections:
                    # Simple relevance scoring based on keyword matches
                    section_lower = section.lower()
                    words = set(re.findall(r'\b\w+\b', query_lower))
                    matches = sum(1 for word in words if word in section_lower)
                    
                    if matches > best_score:
                        best_score = matches
                        best_section = section
                
                # Replace the full response with just the relevant section
                response = best_section
        
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
