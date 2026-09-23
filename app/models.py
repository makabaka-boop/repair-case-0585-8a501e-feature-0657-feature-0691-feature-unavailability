"""Request/response models.

Validation here is what produces HTTP 422 for bad length, out-of-range
values, an illegal L or objective. On any such error FastAPI returns only the
standard ``detail`` payload: no cost and no (partial) segment list.
"""

from typing import Literal

from pydantic import BaseModel, Field, StrictInt, model_validator

SourceName = Literal["A", "B"]
ObjectiveName = Literal["default", "continuity"]
CertaintyName = Literal["A_ONLY", "B_ONLY", "EITHER"]


class PositionCost(BaseModel):
    a: StrictInt = Field(ge=0, le=1_000_000)
    b: StrictInt = Field(ge=0, le=1_000_000)


class GapRequest(BaseModel):
    n: StrictInt = Field(ge=1, le=200_000)
    L: StrictInt = Field(ge=1, le=4_096)
    fee_a: StrictInt = Field(ge=0, le=1_000_000)
    fee_b: StrictInt = Field(ge=0, le=1_000_000)
    objective: ObjectiveName = "default"
    costs: list[PositionCost] = Field(min_length=1, max_length=200_000)

    @model_validator(mode="after")
    def _costs_must_match_n(self) -> "GapRequest":
        if len(self.costs) != self.n:
            raise ValueError("len(costs) must equal n")
        return self


class SegmentOut(BaseModel):
    start: int
    end: int
    source: SourceName


class GapResponse(BaseModel):
    cost: int
    segments: list[SegmentOut]
    # Per position 0..n-1: A_ONLY / B_ONLY mean every minimum-cost cover
    # uses that source there; EITHER means both sources occur in at least one
    # minimum-cost cover. Objective-independent.
    certainty: list[CertaintyName]
