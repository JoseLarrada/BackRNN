from pydantic import BaseModel
from typing import List

class Point(BaseModel):
    x: float
    y: float
    t: float

class StrokeInput(BaseModel):
    stroke: List[Point]
    quality: float
    label: int
