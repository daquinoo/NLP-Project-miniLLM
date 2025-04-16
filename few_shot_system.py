import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class HybridTopicClassifier:
    def __init__(self):
        # Load sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Define topics with descriptions and keywords
        self.topics = {
            "factual_qa": {
                "description": "factual questions and answers about the world",
                "keywords": ["what is", "who was", "when did", "where is", "why does", "how does", "explain", "describe"]
            },
            "creative_writing": {
                "description": "writing stories, poems, essays, scripts, and other creative content",
                "keywords": ["write a", "compose", "create a", "story", "poem", "essay", "creative", "imagine"]
            },
            "reasoning": {
                "description": "logical reasoning, problem-solving, and critical thinking",
                "keywords": ["analyze", "solve", "reason", "think", "evaluate", "compare", "contrast", "examine"]
            },
            "coding": {
                "description": "programming, code generation, and software development",
                "keywords": ["code", "function", "program", "algorithm", "python", "javascript", "implement"]
            },
            "math": {
                "description": "mathematical calculations, equations, and numerical problems",
                "keywords": ["calculate", "compute", "solve", "equation", "formula", "arithmetic", "algebra"]
            },
            "summarization": {
                "description": "summarizing text, articles, and other content",
                "keywords": ["summarize", "summary", "outline", "condense", "shorten", "brief"]
            },
            "translation": {
                "description": "translating text between different languages",
                "keywords": ["translate", "conversion", "convert", "to french", "to spanish", "to english"]
            }
        }
        # Compute embeddings
        self.topic_embeddings = {
            topic: self.model.encode(info["description"]) 
            for topic, info in self.topics.items()
        }
    
    def classify(self, query):
        # Keyword matching
        keyword_scores = {topic: 0 for topic in self.topics}
        query_lower = query.lower()
        for topic, info in self.topics.items():
            for keyword in info["keywords"]:
                if keyword.lower() in query_lower:
                    keyword_scores[topic] += 1
        # Semantic similarity
        query_embedding = self.model.encode(query)
        similarity_scores = {}
        for topic, topic_embedding in self.topic_embeddings.items():
            similarity = cosine_similarity([query_embedding], [topic_embedding])[0][0]
            similarity_scores[topic] = similarity
        # Combine scores (normalized keywords + similarity)
        max_keywords = max(keyword_scores.values()) if max(keyword_scores.values()) > 0 else 1
        combined_scores = {}
        for topic in self.topics:
            # Weight: 40% keywords, 60% similarity
            norm_keyword_score = keyword_scores[topic] / max_keywords
            combined_scores[topic] = 0.4 * norm_keyword_score + 0.6 * similarity_scores[topic]
        # Find  best matching topic
        best_topic = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_topic]
        return best_topic, confidence

# To be implemented
class ExampleDatabase:
    """
    This class will handle loading, storing, and retrieving few-shot examples.
    """
    pass

# To be implemented
class PromptConstructor:
    """
    This class will handle constructing prompts with few-shot examples.
    """
    pass

# Simple test to verify classifier functionality
if __name__ == "__main__":
    classifier = HybridTopicClassifier()
    
    test_queries = [
        "What is the capital of France?",
        "Write a poem about the sea",
        "Analyze the economic impact of climate change",
        "Write a Python function to sort a list",
        "Calculate the area of a circle with radius 5",
        "Summarize the main points of this article",
        "Translate this sentence to Spanish"
    ]
    
    for query in test_queries:
        topic, confidence = classifier.classify(query)
        print(f"Query: {query}")
        print(f"Classified as: {topic} (confidence: {confidence:.4f})")
        print()
