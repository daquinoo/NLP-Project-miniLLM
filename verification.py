import re
import numpy as np

class AnswerVerifier:
    def __init__(self):
        pass
        
    def verify_answer(self, reasoning, final_answer, question_type=None):
        """
        Verify the extracted answer against the reasoning.
        
        Args:
            reasoning (str): The full reasoning text
            final_answer (str): The extracted final answer
            question_type (str): Optional question type for specialized verification
            
        Returns:
            dict: Verification results
        """
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = self.detect_question_type(reasoning)
            
        # Apply appropriate verification
        if question_type == "math":
            return self.verify_math(reasoning, final_answer)
        elif question_type == "multiple_choice":
            return self.verify_multiple_choice(reasoning, final_answer)
        else:
            return self.verify_logical_consistency(reasoning, final_answer)
    
    def detect_question_type(self, text):
        """Detect the type of question based on reasoning content."""
        # Math indicators
        math_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # Basic operations
            r'equation',
            r'formula',
            r'calculate',
            r'solve for',
            r'=\s*\d+'
        ]
        
        # Multiple choice indicators
        mc_patterns = [
            r'option [abcd]',
            r'choice [abcd]',
            r'\([abcd]\)',
            r'multiple choice'
        ]
        
        # Check for math patterns
        for pattern in math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "math"
                
        # Check for multiple choice patterns
        for pattern in mc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "multiple_choice"
                
        # Default to general reasoning
        return "general"
    
    def verify_math(self, reasoning, final_answer):
        """
        Verify mathematical calculations and answers.
        
        Args:
            reasoning (str): The full reasoning text
            final_answer (str): The extracted final answer
            
        Returns:
            dict: Verification results
        """
        # Extract numerical answer from final_answer
        final_number = self._extract_numerical_value(final_answer)
        if final_number is None:
            return {"verified": False, "reason": "No numerical answer found"}
        
        # Extract calculations from reasoning
        calculations = self._extract_calculations(reasoning)
        if not calculations:
            return {"verified": False, "reason": "No calculations found in reasoning"}
        
        # Check if any calculation result matches the final answer
        for calc_expr, calc_result in calculations:
            # Allow for small floating-point differences
            if abs(calc_result - final_number) < 0.001:
                return {
                    "verified": True, 
                    "reason": f"Final answer ({final_number}) matches calculation result from '{calc_expr}'"
                }
        
        # Look for the final calculation
        last_result = calculations[-1][1] if calculations else None
        if last_result is not None and abs(last_result - final_number) < 0.001:
            return {"verified": True, "reason": "Final answer matches the last calculation"}
        
        # No matching calculation found
        return {
            "verified": False, 
            "reason": f"Final answer ({final_number}) doesn't match any calculation results",
            "calculations": [f"{expr} = {result}" for expr, result in calculations]
        }

    def _extract_numerical_value(self, text):
        """Extract a numerical value from text."""
        if not text:
            return None
            
        # Look for numbers with optional decimal point
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        if matches:
            return float(matches[0])
        return None

    def _extract_calculations(self, text):
        """Extract calculations and their results from reasoning."""
        calculations = []
        
        # Find patterns like "3 + 4 = 7" or "3 + 4 equals 7"
        eq_patterns = [
            r'([-+]?\d*\.?\d+\s*[\+\-\*/]\s*[-+]?\d*\.?\d+)\s*=\s*([-+]?\d*\.?\d+)',
            r'([-+]?\d*\.?\d+\s*[\+\-\*/]\s*[-+]?\d*\.?\d+)\s*equals\s*([-+]?\d*\.?\d+)',
            r'([-+]?\d*\.?\d+\s*[\+\-\*/]\s*[-+]?\d*\.?\d+)\s*is\s*([-+]?\d*\.?\d+)'
        ]
        
        for pattern in eq_patterns:
            for match in re.finditer(pattern, text):
                expr = match.group(1).strip()
                result = float(match.group(2).strip())
                
                # Validate by computing the expression
                try:
                    computed = eval(expr)
                    if abs(computed - result) < 0.001:  # Allow small floating point differences
                        calculations.append((expr, result))
                except:
                    pass  # Skip if expression can't be evaluated
        
        return calculations
    
    def verify_multiple_choice(self, reasoning, final_answer):
        """
        Verify multiple choice answers against reasoning.
        
        Args:
            reasoning (str): The full reasoning text
            final_answer (str): The extracted final answer
            
        Returns:
            dict: Verification results
        """
        # Extract the letter answer (A, B, C, D)
        letter_match = re.search(r'\b([ABCD])[\.:]', final_answer) or re.search(r'option\s*([ABCD])', final_answer, re.IGNORECASE)
        option_letter = letter_match.group(1).upper() if letter_match else None
        
        # Look for the chosen option in the reasoning
        if option_letter:
            # Find reasoning supporting this option
            option_pattern = fr'(?:option|choice)\s*{option_letter}[\.:]?\s*([^.!?]*)'
            option_match = re.search(option_pattern, reasoning, re.IGNORECASE)
            
            if option_match:
                option_content = option_match.group(1).strip()
                return {
                    "verified": True,
                    "reason": f"Reasoning supports option {option_letter}: '{option_content}'",
                    "chosen_option": option_letter
                }
        
        # Check for explicit mentions of the chosen answer in reasoning
        reasoning_lower = reasoning.lower()
        final_lower = final_answer.lower()
        
        # Look for phrases like "the answer is X" or "I choose X"
        choice_patterns = [
            r'the answer is ([ABCD])', 
            r'I choose ([ABCD])',
            r'option ([ABCD]) is correct',
            r'selecting ([ABCD])',
            r'choose option ([ABCD])'
        ]
        
        for pattern in choice_patterns:
            pattern_match = re.search(pattern, reasoning_lower)
            if pattern_match:
                reasoned_choice = pattern_match.group(1).upper()
                if option_letter and reasoned_choice == option_letter:
                    return {
                        "verified": True,
                        "reason": f"Reasoning explicitly selects option {reasoned_choice}",
                        "chosen_option": reasoned_choice
                    }
                elif option_letter:
                    return {
                        "verified": False,
                        "reason": f"Reasoning selects {reasoned_choice} but answer gives {option_letter}",
                        "contradiction": True
                    }
        
        return {
            "verified": "uncertain",
            "reason": "Cannot confidently verify multiple choice answer"
        }
    
    def verify_logical_consistency(self, reasoning, final_answer):
        """
        Check for logical consistency between reasoning and answer.
        
        Args:
            reasoning (str): The full reasoning text
            final_answer (str): The extracted final answer
            
        Returns:
            dict: Verification results
        """
        # Check if final answer appears in reasoning
        if final_answer in reasoning:
            return {"verified": True, "reason": "Answer consistent with reasoning"}
        
        # Check if there are contradictions
        contradictions = self._find_contradictions(reasoning, final_answer)
        if contradictions:
            return {
                "verified": False, 
                "reason": "Possible contradiction between reasoning and answer",
                "contradictions": contradictions
            }
        
        # Check if any part of the final answer appears in reasoning
        # Split final answer into meaningful chunks
        answer_parts = self._split_into_chunks(final_answer)
        matching_parts = []
        
        for part in answer_parts:
            if len(part) >= 4 and part in reasoning:  # Only check substantial parts
                matching_parts.append(part)
                
        if matching_parts:
            coverage = sum(len(part) for part in matching_parts) / len(final_answer)
            if coverage > 0.5:  # If more than half is covered
                return {
                    "verified": True, 
                    "reason": f"Answer partially supported by reasoning (coverage: {coverage:.1%})"
                }
        
        # If we can't verify or reject confidently
        return {
            "verified": "uncertain", 
            "reason": "Cannot confidently verify logical consistency"
        }

    def _find_contradictions(self, reasoning, answer):
        """Find potential contradictions between reasoning and answer."""
        contradictions = []
        
        # Look for negative phrases followed by something in the answer
        negation_phrases = ["not", "isn't", "doesn't", "cannot", "never", "false"]
        answer_parts = self._split_into_chunks(answer)
        
        for neg in negation_phrases:
            neg_pattern = f"{neg} ([^.!?]*)"
            matches = re.finditer(neg_pattern, reasoning, re.IGNORECASE)
            
            for match in matches:
                negated_text = match.group(1).strip()
                for part in answer_parts:
                    if len(part) >= 4 and part in negated_text:
                        contradictions.append(f"Answer contains '{part}' but reasoning negates: '{neg} {negated_text}'")
        
        return contradictions

    def _split_into_chunks(self, text):
        """Split text into meaningful chunks for comparison."""
        # First by sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        # If no sentences, split by phrases
        if not sentences:
            phrases = [p.strip() for p in re.split(r'[,;:]', text) if p.strip()]
            if phrases:
                return phrases
        
        # If still nothing, split by words but keep words together
        if not sentences:
            word_groups = []
            words = [w.strip() for w in text.split() if w.strip()]
            
            for i in range(len(words)):
                if i + 1 < len(words):
                    word_groups.append(f"{words[i]} {words[i+1]}")
            
            return word_groups if word_groups else [text]
        
        return sentences
