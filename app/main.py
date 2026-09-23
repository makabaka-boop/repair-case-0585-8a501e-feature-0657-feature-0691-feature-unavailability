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
        unavailable_a = unavailable_b = ()
    else:
        unavailable_a = tuple(
            (w.start, w.end) for w in req.unavailable.a)
        unavailable_b = tuple(
            (w.start, w.end) for w in req.unavailable.b)
    try:
        result = solve(
            n=req.n,
            costs=[(p.a, p.b) for p in req.costs],
            fee_a=req.fee_a,
            fee_b=req.fee_b,
            max_len=req.L,
            objective=req.objective,
            unavailable_a=unavailable_a,
            unavailable_b=unavailable_b,
        )
    except NoFeasibleRepair as exc:
        # Body carries only ``detail``; no partial cost, segments or
        # certainty may leave the solver for an infeasible instance.
        raise HTTPException(status_code=409, detail="NO_FEASIBLE_REPAIR") \
            from exc
    return GapResponse(
        cost=result.cost,
        segments=[
            {"start": seg.start, "end": seg.end, "source": seg.source}
            for seg in result.segments
        ],
        certainty=list(result.certainty),
    )
