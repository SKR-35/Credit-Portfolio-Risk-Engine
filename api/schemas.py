from typing import Dict
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: Dict[str, float]