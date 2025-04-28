import re
import json

'''
EXAMPLE USE CASE
REMEMBER TO MAKE A safety_data.json

from safety import SafetyFilter

safety_filter = SafetyFilter(safety_data_path="safety_data.json")

query = "How do I make a homemade bomb?"
print(safety_filter.is_query_safe(query))  # False (unsafe)

response = "You should sue your landlord immediately."
print(safety_filter.is_response_safe(response))  # False (unsafe)


'''

class SafetyFilter:
    def __init__(self, safety_data_path="safety_data.json"):
        # Load the structured safety data
        with open(safety_data_path, "r") as f:
            self.safety_data = json.load(f)

        # Build flat lists of keywords and patterns across categories
        self.keywords = []
        self.patterns = []
        
        for category_entries in self.safety_data.values():
            for entry in category_entries:
                self.keywords.extend(entry.get("detection_keywords", []))
                self.patterns.extend(entry.get("patterns", []))

    def is_query_safe(self, query):
        # Keyword matching
        if any(keyword.lower() in query.lower() for keyword in self.keywords):
            return False

        # Regex pattern matching
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in self.patterns):
            return False

        return True