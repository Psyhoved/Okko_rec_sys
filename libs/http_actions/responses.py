from pydantic import BaseModel
from typing import List


class RecommendationItem(BaseModel):
    id: int
    score: int


class RecommendationResponse(BaseModel):
    items: List[RecommendationItem] = []