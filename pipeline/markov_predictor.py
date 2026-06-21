from collections import defaultdict
from typing import List, Dict

class MarkovPredictor:
    """
    Builds a baseline Transition Matrix P(next_state | current_state) 
    using the temporal sequence.
    """
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(dict)
        self.total_transitions = 0

    def build_matrix(self, sequences: List[List[str]]):
        for seq in sequences:
            for i in range(len(seq) - 1):
                curr_state = seq[i]
                next_state = seq[i+1]
                self.transition_counts[curr_state][next_state] += 1
                self.total_transitions += 1
                
        # Normalize to probabilities
        for curr_state, next_states in self.transition_counts.items():
            total = sum(next_states.values())
            for next_state, count in next_states.items():
                self.transition_probs[curr_state][next_state] = count / total

    def top_k_predictions(self, current_state: str, k: int = 3) -> List[dict]:
        if current_state not in self.transition_probs:
            return []
            
        next_states = self.transition_probs[current_state]
        sorted_states = sorted(next_states.items(), key=lambda item: item[1], reverse=True)
        return [{"state": state, "probability": round(prob, 2)} for state, prob in sorted_states[:k]]
