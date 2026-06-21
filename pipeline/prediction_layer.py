from typing import List
from schemas.cti_schema import CTIEvent
from datetime import datetime

class TemporalSequencer:
    """
    Sorts unstructured retrieved events into a chronologically ordered sequence.
    """
    def sequence_events(self, retrieved_event_ids: List[str], all_events_dict: dict) -> List[str]:
        event_objects = []
        for eid in retrieved_event_ids:
            evt = all_events_dict.get(eid)
            if evt:
                event_objects.append(evt)
        
        def get_date(e: CTIEvent):
            if e.date:
                try:
                    return datetime.strptime(e.date, "%Y-%m-%d")
                except ValueError:
                    pass
            return datetime.min

        event_objects.sort(key=lambda e: (get_date(e), e.event_id))
        return [f"evt_{e.event_id}" for e in event_objects]
