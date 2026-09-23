"""FastAPI application for the telemetry-gap repair DP."""

from fastapi import FastAPI, HTTPException

from .models import GapRequest, GapResponse
from .solver import NoFeasibleRepair, solve

app = FastAPI(
    title="Telemetry Gap Repair",
    summary="Minimum-cost cover of a telemetry gap by A/B repair segments",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=GapResponse)
def solve_gap(req: GapRequest) -> GapResponse:
    if req.unavailable is None:
        unavailable = None
    else:
        unavailable = (
            [(w.start, w.end) for w in req.unavailable.A],
            [(w.start, w.end) for w in req.unavailable.B],
        )
    try:
        result = solve(
            n=req.n,
            costs=[(p.a, p.b) for p in req.costs],
            fee_a=req.fee_a,
            fee_b=req.fee_b,
            max_len=req.L,
            objective=req.objective,
            unavailable=unavailable,
        )
    except NoFeasibleRepair:
        # Only the standard detail field: no cost, segments or certainty may
        # leak from a partial solution.
        raise HTTPException(status_code=409, detail="NO_FEASIBLE_REPAIR")
    return GapResponse(
        cost=result.cost,
        segments=[
            {"start": seg.start, "end": seg.end, "source": seg.source}
            for seg in result.segments
        ],
        certainty=list(result.certainty),
    )
