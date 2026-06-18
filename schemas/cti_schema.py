from pydantic import BaseModel, Field
from typing import List, Optional

class CTIAttribute(BaseModel):
    id: int = Field(..., description="Unique attribute ID")
    category: str = Field(..., description="Category of the IoC (e.g., Network activity)")
    type: str = Field(..., description="Type of the IoC (e.g., url, md5)")
    value: str = Field(..., description="The IoC value")
    comment: Optional[str] = Field(None, description="Additional context or tag")

class CTIEvent(BaseModel):
    event_id: int = Field(..., description="Event ID")
    source_file: str = Field(..., description="The XML file this event came from")
    event_type: str = Field(..., description="ReportEvent or MalwareEvent")
    report_year: int = Field(..., description="The year extracted from the file name")
    date: str = Field(..., description="Event date (YYYY-MM-DD)")
    info: str = Field(..., description="Contextual info or report title")
    attributes: List[CTIAttribute] = Field(default_factory=list, description="List of associated attributes")

    @property
    def narrative(self) -> str:
        # Representation B: Narrative string
        event_desc = "Malware" if self.event_type == "MalwareEvent" else "Report"
        base = f"CTI {event_desc} from {self.report_year} titled '{self.info}'. "
        base += f"Observed on {self.date} in source {self.source_file}. "
        
        categories = set(a.category for a in self.attributes if a.category)
        if categories:
            base += f"This event contains behavioral attributes related to: {', '.join(categories)}."
        else:
            base += "This event contains no categorized attributes."
            
        return base
