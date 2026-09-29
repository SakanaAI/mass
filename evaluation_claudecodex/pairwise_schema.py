from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PairwiseJudgeResult(BaseModel):
    winner: Literal["A", "B", "tie"]
    rationale: str = Field(min_length=1)
