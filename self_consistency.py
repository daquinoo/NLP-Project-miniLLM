import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SelfConsistencyFramework:
    def __init__(self, model, tokenizer, prompt_constructor):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_constructor = prompt_constructor
        # TF-IDF vectorizer for clustering similar answers
        self.vectorizer = None  # Will be initialized on first use
    
    def generate_multiple_responses(self, query, topic=None, num_samples=5, 
                                   temperature_range=(0.7, 0.9), max_new_tokens=300):
        """
        Generate multiple diverse reasoning paths for the same input.
        
        Args:
            query (str): The query to process
            topic (str): Optional pre-classified topic
            num_samples (int): Number of different reasoning paths to generate
            temperature_range (tuple): Range of temperatures to use for diversity
            max_new_tokens (int): Maximum new tokens to generate per sample
            
        Returns:
            list: List of generated reasoning paths
        """
        # Use chain-of-thought prompt construction
        prompt = self.prompt_constructor.construct_prompt(
            query, 
            topic=topic, 
            use_cot=True
        )
        
        responses = []
        for i in range(num_samples):
            # Scale temperature linearly across the range for diverse samples
            temp = temperature_range[0] + (i / max(1, num_samples-1)) * (temperature_range[1] - temperature_range[0])
            
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate with current temperature
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.95,  # Use nucleus sampling
                    no_repeat_ngram_size=3  # Prevent repetitive outputs
                )
            
            # Decode and extract just the response (not the prompt)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.replace(prompt, "").strip()
            responses.append(response)
            
        return responses
    
    def cluster_similar_answers(self, answers, similarity_threshold=0.85):
        """
        Group semantically similar answers together using TF-IDF.
        
        Args:
            answers (list): List of extracted final answers
            similarity_threshold (float): Threshold for considering answers similar
            
        Returns:
            list: List of clusters, where each cluster is a list of similar answers
        """
        if not answers or len(answers) <= 1:
            return [answers] if answers else []
        
        # Initialize vectorizer on first use
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        
        # Compute TF-IDF vectors
        try:
            vectors = self.vectorizer.fit_transform(answers)
        except:
            # Fallback in case of vectorization errors
            return [answers]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(vectors)
        
        # Cluster similar answers
        clusters = []
        used_indices = set()
        
        for i in range(len(answers)):
            if i in used_indices:
                continue
                
            # Start a new cluster with this answer
            cluster = [i]
            used_indices.add(i)
            
            # Find all similar answers
            for j in range(len(answers)):
                if j in used_indices:
                    continue
                    
                if similarities[i, j] >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append([answers[idx] for idx in cluster])
        
        # Sort clusters by size (largest first)
        return sorted(clusters, key=len, reverse=True)

    def get_majority_answer(self, reasoning_paths):
        """
        Extract answers from multiple reasoning paths and find the most consistent one.
        
        Args:
            reasoning_paths (list): List of reasoning paths from the model
            
        Returns:
            tuple: (majority_answer, confidence_score, all_answers)
        """
        # Extract final answers from each reasoning path
        extracted_answers = [
            self.prompt_constructor.extract_final_answer(path)
            for path in reasoning_paths
        ]
        
        # Check if we have any non-empty answers
        valid_answers = [a for a in extracted_answers if a and a != "No answer provided"]
        if not valid_answers:
            return "No consistent answer found", 0.0, extracted_answers
        
        # Group similar answers
        clusters = self.cluster_similar_answers(valid_answers)
        
        if not clusters:
            return valid_answers[0], 1/len(valid_answers), valid_answers
        
        # The largest cluster is the majority answer group
        majority_cluster = clusters[0]
        
        # Choose the shortest answer from the majority cluster as the canonical answer
        # (often the most concise is clearest)
        majority_answer = min(majority_cluster, key=len) if majority_cluster else valid_answers[0]
        
        # Calculate confidence as size of largest cluster divided by total valid answers
        confidence = len(majority_cluster) / len(valid_answers)
        
        return majority_answer, confidence, extracted_answers
    
    def get_confidence_level(self, confidence_score):
        """Convert numerical confidence to descriptive level."""
        if confidence_score >= 0.8:
            return "High"
        elif confidence_score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def generate_with_self_consistency(self, query, topic=None, num_samples=5):
        """
        Complete pipeline for self-consistency based answer generation.
        
        Args:
            query (str): User query
            topic (str): Optional pre-classified topic
            num_samples (int): Number of reasoning paths to generate
            
        Returns:
            dict: Results containing final answer, confidence, and supporting info
        """
        # Generate multiple reasoning paths
        reasoning_paths = self.generate_multiple_responses(query, topic, num_samples)
        
        # Extract and find majority answer
        majority_answer, confidence_score, all_answers = self.get_majority_answer(reasoning_paths)
        
        # Determine confidence level
        confidence_level = self.get_confidence_level(confidence_score)
        
        return {
            "final_answer": majority_answer,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "reasoning_paths": reasoning_paths,
            "extracted_answers": all_answers
        }
