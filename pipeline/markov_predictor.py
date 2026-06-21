import os
import sys
from collections import defaultdict
from typing import List
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

class GlobalMarkovPredictor:
    """
    Precomputes a mathematically rigorous Transition Matrix across the entire CTI corpus.
    Uses Technique-States extracted via the V2 Classifier.
    """
    def __init__(self, base_dir: str):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(dict)
        self.total_transitions = 0
        self.base_dir = base_dir
        self.classifier = DeterministicClassifierV2(base_dir)
        self._build_global_matrix()

    def _get_date(self, e):
        if e.date:
            try:
                return datetime.strptime(e.date, "%Y-%m-%d")
            except ValueError:
                pass
        return datetime.min

    def _build_global_matrix(self):
        xml_path = os.path.join(self.base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
        events = parse_xml_file(xml_path)
        sorted_events = sorted(events, key=lambda x: (self._get_date(x), x.event_id))
        
        technique_sequence = []
        for e in sorted_events:
            res = self.classifier.classify_event(e)
            if res['techniques']:
                technique_sequence.append(res['techniques'][0]['id'])
                
        for i in range(len(technique_sequence) - 1):
            curr_state = technique_sequence[i]
            next_state = technique_sequence[i+1]
            self.transition_counts[curr_state][next_state] += 1
            self.total_transitions += 1
            
        for curr_state, next_states in self.transition_counts.items():
            total = sum(next_states.values())
            for next_state, count in next_states.items():
                self.transition_probs[curr_state][next_state] = count / total

    def top_k_predictions(self, current_state: str, k: int = 5) -> List[dict]:
        if current_state not in self.transition_probs:
            return []
            
        next_states = self.transition_probs[current_state]
        sorted_states = sorted(next_states.items(), key=lambda item: item[1], reverse=True)
        return [{"state": state, "probability": round(prob, 2)} for state, prob in sorted_states[:k]]
