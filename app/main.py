"""FastAPI application for the telemetry-gap repair DP."""

from fastapi import FastAPI

from .models import GapRequest, GapResponse
from .solver import solve

app = FastAPI(
    title="Telemetry Gap Repair",
    summary="Minimum-cost cover of a telemetry gap by A/B repair segments",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=GapResponse)
def solve_gap(req: GapRequest) -> GapResponse:
    result = solve(
        n=req.n,
        costs=[(p.a, p.b) for p in req.costs],
        fee_a=req.fee_a,
        fee_b=req.fee_b,
        max_len=req.L,
        objective=req.objective,
    )
    return GapResponse(
        cost=result.cost,
        segments=[
            {"start": seg.start, "end": seg.end, "source": seg.source}
            for seg in result.segments
        ],
        certainty=list(result.certainty),
    )
