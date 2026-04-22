from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class IntensityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerbalTask(BaseModel):
    id: str
    task: str
    context: str
    question: str
    options: List[str]
    label: str
    rationale: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    results: Optional[Dict[str, Any]] = Field(default_factory=dict)
    validation: Optional[Dict[str, Any]] = Field(default_factory=dict)
