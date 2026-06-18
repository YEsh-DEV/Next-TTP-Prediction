from pydantic import BaseModel, Field
from typing import List, Optional

class CTIAttribute(BaseModel):
    id: int = Field(..., description="Unique attribute ID")
    category: str = Field(..., description="Category of the IoC (e.g., Network activity)")
    type: str = Field(..., description="Type of the IoC (e.g., url, md5)")
    value: str = Field(..., description="The IoC value")
    comment: Optional[str] = Field(None, description="Additional context or tag")

class CTIEvent(BaseModel):
    id: int = Field(..., description="Event ID")
    date: str = Field(..., description="Event date (YYYY-MM-DD)")
    info: str = Field(..., description="Contextual info or report title")
    attributes: List[CTIAttribute] = Field(default_factory=list, description="List of associated attributes")
