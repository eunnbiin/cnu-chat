from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class SourceInfo(BaseModel):
    title: str
    url: str = ""
    similarity: float
    preview: str

class QuestionResponse(BaseModel):
    answer: str
    category: str
    category_id: int
    confidence: float
    sources: List[SourceInfo] = []
    context_used: bool
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.now)