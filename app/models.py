"""Request/response models.

Validation here is what produces HTTP 422 for bad length, out-of-range
values, an illegal L or objective. On any such error FastAPI returns only the
standard ``detail`` payload: no cost and no (partial) segment list.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictInt, model_validator

SourceName = Literal["A", "B"]
ObjectiveName = Literal["default", "continuity"]
CertaintyName = Literal["A_ONLY", "B_ONLY", "EITHER"]


class PositionCost(BaseModel):
    a: StrictInt = Field(ge=0, le=1_000_000)
    b: StrictInt = Field(ge=0, le=1_000_000)


class UnavailableWindow(BaseModel):
    """A half-open maintenance window ``[start, end)`` for one source."""

    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(ge=0)


class Unavailable(BaseModel):
    """Per-source maintenance windows, keyed by position.

    Each list is validated on the parent request: endpoints inside
    ``[0, n]``, strictly increasing starts, and no overlap or touching.
    """

    a: list[UnavailableWindow] = Field(default_factory=list)
    b: list[UnavailableWindow] = Field(default_factory=list)


class GapRequest(BaseModel):
    n: StrictInt = Field(ge=1, le=200_000)
    L: StrictInt = Field(ge=1, le=4_096)
    fee_a: StrictInt = Field(ge=0, le=1_000_000)
    fee_b: StrictInt = Field(ge=0, le=1_000_000)
    objective: ObjectiveName = "default"
    costs: list[PositionCost] = Field(min_length=1, max_length=200_000)
    unavailable: Optional[Unavailable] = None

    @model_validator(mode="after")
    def _costs_must_match_n(self) -> "GapRequest":
        if len(self.costs) != self.n:
            raise ValueError("len(costs) must equal n")
        self._validate_windows("a")
        self._validate_windows("b")
        return self

    def _validate_windows(self, name: str) -> None:
        if self.unavailable is None:
            return
        windows = getattr(self.unavailable, name)
        # -1 lets a first window legally start at 0; afterwards the next
        # start must be strictly greater than the previous end, rejecting
        # both overlap and touching (half-open) intervals.
        prev_end = -1
        for window in windows:
            if not (0 <= window.start < window.end <= self.n):
                raise ValueError(
                    f"unavailable.{name} endpoints must satisfy "
                    "0 <= start < end <= n"
                )
            if window.start <= prev_end:
                raise ValueError(
                    f"unavailable.{name} windows must be strictly "
                    "increasing and must not overlap or touch"
                )
            prev_end = window.end


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
