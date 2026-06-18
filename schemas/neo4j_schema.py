from pydantic import BaseModel, Field
from typing import Optional

class APTGroupSchema(BaseModel):
    name: str = Field(..., description="Name of the APT Group")

class SoftwareSchema(BaseModel):
    id: str = Field(..., description="MITRE Software ID (e.g., S0031)")

class TechniqueSchema(BaseModel):
    id: str = Field(..., description="MITRE Technique ID (e.g., T1059)")
    name: Optional[str] = Field(None, description="Name of the technique")
    description: Optional[str] = Field(None, description="Description of the technique")
    mitigation_steps: Optional[str] = Field(None, description="Steps to mitigate")

class TacticSchema(BaseModel):
    name: str = Field(..., description="MITRE Tactic Name (e.g., Execution)")
