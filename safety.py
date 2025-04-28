import re
import json

class SafetyFilter:
    def __init__(self, safety_data_path="safety_data.json"):
        with open(safety_data_path, "r") as f:
            self.safety_data = json.load(f)

        self.keywords = []
        self.patterns = []
        self.responses = []

        for category_entries in self.safety_data.values():
            for entry in category_entries:
                self.keywords.extend(entry.get("detection_keywords", []))
                self.patterns.extend(entry.get("patterns", []))
                self.responses.append((
                    entry.get("detection_keywords", []), 
                    entry.get("patterns", []), 
                    entry.get("refusal_response", "")
                ))

    def check_query(self, query):
        query_lower = query.lower()

        # First, check keyword matches
        for keywords, patterns, response in self.responses:
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return False, response

        # Then, check pattern matches
        for keywords, patterns, response in self.responses:
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return False, response

        return True, None

    def is_query_safe(self, query):
        safe, _ = self.check_query(query)
        return safe

    def get_refusal_response(self, query):
        safe, response = self.check_query(query)
        if not safe:
            return response
        return None
