"""Tests for per-source maintenance windows (``unavailable``).

* An oracle enumerates every legal segmentation/source assignment and
  filters out any segment intersecting that source's maintenance windows;
  it cross-checks both objectives, the certainty union and the infeasible
  verdict on small exhaustive instances and randomized medium ones.
* Window endpoints (0, n, segment boundaries), one source fully down, both
  sources down at once, touching-interval rejection and old-request
  regression are pinned explicitly at the API layer.
* A 409 response only ever carries ``{"detail": "NO_FEASIBLE_REPAIR"}``.
"""

import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    SOURCE_A,
    SOURCE_B,
    NoFeasibleRepair,
    Segment,
    solve,
)

client = TestClient(app)


# --------------------------------------------------------------------------
# Reference implementations (window-aware)
# --------------------------------------------------------------------------


def enumerate_plans(n, max_len, unavailable=(set(), set())):
    """Enumerate every legal segmentation and A/B source assignment.

    A source-s segment is skipped as soon as it touches a position of one
    of that source's windows, so only availability-respecting plans are
    yielded. No position/window product is built.
    """
    blocked_a, blocked_b = unavailable
    blocked = (blocked_a, blocked_b)

    def dfs(start, plan):
        if start == n:
            yield tuple(plan)
            return
        upper = min(n, start + max_len)
        touched = [set(), set()]
        for end in range(start + 1, upper + 1):
            new_pos = end - 1
            for s in (SOURCE_A, SOURCE_B):
                if new_pos in blocked[s]:
                    touched[s].add(new_pos)
            for source in (SOURCE_A, SOURCE_B):
                if not touched[source]:
                    yield from dfs(end, plan + [(start, end, source)])

    yield from dfs(0, [])


def plan_cost(plan, pref, fees):
    total = 0
    for start, end, source in plan:
        total += fees[source] + pref[source][end] - pref[source][start]
    return total


def default_plan_key(plan, pref, fees):
    if not plan:
        return ()
    start, _end, source = plan[-1]
    return (
        plan_cost(plan, pref, fees),
        len(plan),
        start,
        source,
        default_plan_key(plan[:-1], pref, fees),
    )


def continuity_plan_key(plan, pref, fees):
    if not plan:
        return ()
    start, _end, source = plan[-1]
    prefix = plan[:-1]
    switches = sum(
        left[2] != right[2] for left, right in zip(plan, plan[1:])
    )
    return (
        plan_cost(plan, pref, fees),
        switches,
        len(plan),
        start,
        source,
        continuity_plan_key(prefix, pref, fees),
    )


def make_pref(n, costs):
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]
    return pref


def constrained_oracle(n, costs, fa, fb, max_len, win_a, win_b, objective):
    """Pick the winner directly from all availability-respecting plans."""
    fees = (fa, fb)
    pref = make_pref(n, costs)
    blocked = (windows_to_blocked(n, win_a), windows_to_blocked(n, win_b))
    plans = list(enumerate_plans(n, max_len, blocked))
    if not plans:
        return None
    key_func = (
        continuity_plan_key if objective == "continuity"
        else default_plan_key
    )
    best_plan = min(plans, key=lambda plan: key_func(plan, pref, fees))
    return (
        plan_cost(best_plan, pref, fees),
        [(start, end, "AB"[source]) for start, end, source in best_plan],
    )


def certainty_oracle_constrained(n, costs, fa, fb, max_len, win_a, win_b):
    """Union of per-position sources over all minimum-cost feasible plans."""
    fees = (fa, fb)
    pref = make_pref(n, costs)
    blocked = (windows_to_blocked(n, win_a), windows_to_blocked(n, win_b))
    plans = list(enumerate_plans(n, max_len, blocked))
    assert plans, "oracle called on an infeasible instance"
    optimum = min(plan_cost(plan, pref, fees) for plan in plans)
    possible = [[False, False] for _ in range(n)]
    for plan in plans:
        if plan_cost(plan, pref, fees) != optimum:
            continue
        for start, end, source in plan:
            for k in range(start, end):
                possible[k][source] = True
    return [
        "EITHER" if a and b else "A_ONLY" if a else "B_ONLY"
        for a, b in possible
    ]


def windows_to_blocked(n, windows):
    blocked = set()
    for start, end in windows:
        blocked.update(range(start, end))
    return blocked


def random_windows(rng, n):
    """A sorted, non-overlapping, non-touching random window list."""
    windows = []
    pos = rng.randrange(0, 3)  # sometimes leave the first positions open
    while pos < n:
        if rng.random() < 0.45:
            length = rng.randrange(1, min(n - pos, 3) + 1)
            windows.append((pos, pos + length))
            pos += length + rng.randrange(1, 3)  # gap >= 1: never touch
        else:
            pos += 1
    return windows


def assert_plan_respects_windows(segs, win_a, win_b):
    windows = {"A": win_a, "B": win_b}
    for seg in segs:
        for start, end in windows[seg.source]:
            # Half-open intersection test.
            assert not (seg.start < end and start < seg.end), (
                f"segment {seg} intersects window [{start},{end})")


def assert_segments_sound(segs, n, max_len):
    assert segs[0].start == 0
    assert segs[-1].end == n
    for a, b in zip(segs, segs[1:]):
        assert a.end == b.start
    for seg in segs:
        assert 1 <= seg.end - seg.start <= max_len


# --------------------------------------------------------------------------
# Exhaustive oracle cross-checks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["default", "continuity"])
@pytest.mark.parametrize("n", range(1, 8))
def test_exhaustive_windows_oracle(n, objective):
    # Sweep a deterministic variety of random windows; the oracle is the
    # authority on feasibility and on the exact winning plan.
    rng = random.Random(2000 + n)
    for trial in range(10):
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 6), rng.randrange(0, 6))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 5), rng.randrange(0, 5)
        win_a = random_windows(rng, n)
        win_b = random_windows(rng, n)

        oracle_result = constrained_oracle(
            n, costs, fa, fb, L, win_a, win_b, objective)
        if oracle_result is None:
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable_a=tuple(win_a),
                      unavailable_b=tuple(win_b))
            continue

        sol = solve(n, costs, fa, fb, L, objective=objective,
                    unavailable_a=tuple(win_a), unavailable_b=tuple(win_b))
        ocost, osegs = oracle_result
        assert sol.cost == ocost
        got = [(s.start, s.end, s.source) for s in sol.segments]
        assert got == osegs
        assert_segments_sound(list(sol.segments), n, L)
        assert_plan_respects_windows(list(sol.segments), win_a, win_b)

        expected_certainty = certainty_oracle_constrained(
            n, costs, fa, fb, L, win_a, win_b)
        assert list(sol.certainty) == expected_certainty

        # Certainty must remain identical under the other objective.
        other = solve(
            n, costs, fa, fb, L,
            objective="continuity" if objective == "default" else "default",
            unavailable_a=tuple(win_a), unavailable_b=tuple(win_b))
        assert list(other.certainty) == expected_certainty


def test_exhaustive_single_source_only():
    # Source A is fully down everywhere: every feasible plan uses B only,
    # for every small n/L, and certainty is B_ONLY throughout.
    rng = random.Random(4242)
    for n in range(1, 7):
        for L in range(1, n + 1):
            costs = [(rng.randrange(0, 5), rng.randrange(0, 5))
                     for _ in range(n)]
            fa, fb = rng.randrange(0, 4), rng.randrange(0, 4)
            win_a = [(0, n)]
            for objective in ("default", "continuity"):
                sol = solve(n, costs, fa, fb, L, objective=objective,
                            unavailable_a=((0, n),))
                assert all(s.source == "B" for s in sol.segments)
                assert list(sol.certainty) == ["B_ONLY"] * n
                ocost, osegs = constrained_oracle(
                    n, costs, fa, fb, L, win_a, (), objective)
                assert sol.cost == ocost
                assert [(s.start, s.end, s.source)
                        for s in sol.segments] == osegs


def test_exhaustive_both_sources_down_at_one_position():
    # A single position unavailable on both sources is always infeasible.
    for n in range(1, 6):
        for down in range(n):
            costs = [(0, 0)] * n
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, 0, 0, n,
                      unavailable_a=((down, down + 1),),
                      unavailable_b=((down, down + 1),))


def test_window_endpoint_boundaries():
    # Windows abut segment boundaries; half-open semantics mean a segment
    # may end exactly where a window starts and start exactly where one
    # ends. n=4, L=2: A windows [1,2) and [2,3) are legal (they touch
    # each other only across sources is irrelevant) -- here they are on
    # the same source but [1,2),[2,3) DO touch, so use A [1,2), B [2,3).
    n, L, costs, fa, fb = 4, 2, [(0, 0)] * 4, 0, 0
    win_a, win_b = ((1, 2),), ((2, 3),)
    for objective in ("default", "continuity"):
        sol = solve(n, costs, fa, fb, L, objective=objective,
                    unavailable_a=win_a, unavailable_b=win_b)
        ocost, osegs = constrained_oracle(
            n, costs, fa, fb, L, list(win_a), list(win_b), objective)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source)
                for s in sol.segments] == osegs
        assert_plan_respects_windows(list(sol.segments),
                                     list(win_a), list(win_b))


def test_window_spanning_n_boundary():
    # A window covering the tail [2, n) forces the last segment of that
    # source to end at 2; a window starting at 0 forces the first to start
    # after it. Cross-check against the enumerator on zero costs where the
    # tie adjudication differs between objectives.
    n, L = 4, 3
    costs = [(0, 0)] * n
    win_a, win_b = ((0, 1),), ((2, 4),)
    for objective in ("default", "continuity"):
        sol = solve(n, costs, 0, 0, L, objective=objective,
                    unavailable_a=win_a, unavailable_b=win_b)
        ocost, osegs = constrained_oracle(
            n, costs, 0, 0, L, list(win_a), list(win_b), objective)
        assert [(s.start, s.end, s.source)
                for s in sol.segments] == osegs
        assert list(sol.certainty) == certainty_oracle_constrained(
            n, costs, 0, 0, L, list(win_a), list(win_b))


def test_medium_random_constrained():
    rng = random.Random(7777)
    for _ in range(40):
        n = rng.randrange(8, 22)
        L = rng.randrange(1, min(n, 7) + 1)
        costs = [(rng.randrange(0, 20), rng.randrange(0, 20))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 8), rng.randrange(0, 8)
        win_a = random_windows(rng, n)
        win_b = random_windows(rng, n)
        oracle_result_default = constrained_oracle(
            n, costs, fa, fb, L, win_a, win_b, "default")
        oracle_result_cont = constrained_oracle(
            n, costs, fa, fb, L, win_a, win_b, "continuity")
        if oracle_result_default is None:
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L,
                      unavailable_a=tuple(win_a),
                      unavailable_b=tuple(win_b))
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L, objective="continuity",
                      unavailable_a=tuple(win_a),
                      unavailable_b=tuple(win_b))
            continue
        sol = solve(n, costs, fa, fb, L,
                    unavailable_a=tuple(win_a),
                    unavailable_b=tuple(win_b))
        assert sol.cost == oracle_result_default[0]
        assert [(s.start, s.end, s.source)
                for s in sol.segments] == oracle_result_default[1]
        cont = solve(n, costs, fa, fb, L, objective="continuity",
                     unavailable_a=tuple(win_a),
                     unavailable_b=tuple(win_b))
        assert cont.cost == oracle_result_cont[0]
        assert [(s.start, s.end, s.source)
                for s in cont.segments] == oracle_result_cont[1]
        assert sol.cost == cont.cost
        assert list(sol.certainty) == list(cont.certainty)
        assert list(sol.certainty) == certainty_oracle_constrained(
            n, costs, fa, fb, L, win_a, win_b)
        for sol_obj in (sol, cont):
            assert_segments_sound(list(sol_obj.segments), n, L)
            assert_plan_respects_windows(
                list(sol_obj.segments), win_a, win_b)


def test_constrained_plan_replay_cost():
    # The reported cost must be directly recomputable from the definition
    # while respecting windows.
    rng = random.Random(909)
    for _ in range(20):
        n = rng.randrange(5, 25)
        L = rng.randrange(1, min(n, 6) + 1)
        costs = [(rng.randrange(0, 50), rng.randrange(0, 50))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 10), rng.randrange(0, 10)
        win_a, win_b = random_windows(rng, n), random_windows(rng, n)
        try:
            sol = solve(n, costs, fa, fb, L,
                        unavailable_a=tuple(win_a),
                        unavailable_b=tuple(win_b))
        except NoFeasibleRepair:
            continue
        total = 0
        for seg in sol.segments:
            s = 0 if seg.source == "A" else 1
            total += (fa if s == 0 else fb) + sum(
                costs[k][s] for k in range(seg.start, seg.end))
        assert total == sol.cost


# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------


def _payload(n=5, **overrides):
    p = {
        "n": n,
        "L": 3,
        "fee_a": 1,
        "fee_b": 2,
        "costs": [{"a": k, "b": k + 1} for k in range(n)],
    }
    p.update(overrides)
    return p


def test_api_unavailable_happy_path():
    payload = _payload(
        n=4, L=2, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 4,
        unavailable={"a": [{"start": 1, "end": 2}],
                     "b": [{"start": 2, "end": 3}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    for seg in data["segments"]:
        windows = (payload["unavailable"]["a"] if seg["source"] == "A"
                   else payload["unavailable"]["b"])
        for w in windows:
            assert not (seg["start"] < w["end"] and w["start"] < seg["end"])


def test_api_omitted_and_empty_unavailable_compatible():
    base = _payload()
    omitted = client.post("/solve", json=base)
    nulled = client.post("/solve", json={**base, "unavailable": None})
    empty = client.post("/solve", json={
        **base, "unavailable": {"a": [], "b": []}})
    only_a = client.post("/solve", json={
        **base, "unavailable": {"a": []}})
    assert omitted.status_code == 200
    for resp in (nulled, empty, only_a):
        assert resp.status_code == 200
        assert resp.json() == omitted.json()


def test_api_409_body_has_only_detail():
    # Position 2 is down on both sources.
    payload = _payload(
        unavailable={"a": [{"start": 2, "end": 3}],
                     "b": [{"start": 2, "end": 3}]})
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_409_single_source_gap_larger_than_L():
    # A is entirely down; B has a maintenance run longer than L, so B alone
    # cannot bridge it -> infeasible on either objective.
    payload = _payload(
        n=6, L=2,
        costs=[{"a": 0, "b": 0}] * 6,
        unavailable={"a": [{"start": 0, "end": 6}],
                     "b": [{"start": 1, "end": 4}]},
    )
    for body in (payload, {**payload, "objective": "continuity"}):
        resp = client.post("/solve", json=body)
        assert resp.status_code == 409
        assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_409_both_sources_down_simultaneously():
    payload = _payload(
        n=3, L=3,
        costs=[{"a": 0, "b": 0}] * 3,
        unavailable={"a": [{"start": 0, "end": 2}],
                     "b": [{"start": 1, "end": 3}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_409_window_endpoint_semantics():
    # Half-open endpoint n: a window [3,3) is illegal, but a window ending
    # exactly at n covers the last position. Here A is down on [1,3) while
    # B's window [3,n) forbids the last B segment -- however B's [1,3) and
    # a final A... instead check the legal boundary case directly: A down
    # [1,3), B free, the optimum is A[0,1)+B[1,3) with the window edge at
    # endpoint 3 untouched.
    payload = _payload(
        n=3, L=3, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 3,
        unavailable={"a": [{"start": 1, "end": 3}],
                     "b": [{"start": 3, "end": 3}]},
    )
    # start == end must stay a validation error even at endpoint n.
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 422
    assert set(resp.json()) == {"detail"}

    # A window ending exactly at n covers the final positions: B down
    # [1,3) means no B segment may cover position 1 or 2, so the only
    # feasible cover must use A there.
    payload2 = _payload(
        n=3, L=3, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 3,
        unavailable={"b": [{"start": 1, "end": 3}]},
    )
    resp2 = client.post("/solve", json=payload2)
    assert resp2.status_code == 200
    segs = resp2.json()["segments"]
    for seg in segs:
        if seg["source"] == "B":
            assert seg["end"] <= 1


@pytest.mark.parametrize("unavailable", [
    # start < 0 / end > n
    {"a": [{"start": -1, "end": 2}]},
    {"b": [{"start": 0, "end": 6}]},
    # start == end
    {"a": [{"start": 2, "end": 2}]},
    # start > end
    {"a": [{"start": 3, "end": 2}]},
    # overlap
    {"a": [{"start": 0, "end": 3}, {"start": 2, "end": 4}]},
    # touching half-open intervals
    {"a": [{"start": 0, "end": 2}, {"start": 2, "end": 4}]},
    {"b": [{"start": 1, "end": 2}, {"start": 2, "end": 3}]},
    # unsorted
    {"a": [{"start": 3, "end": 4}, {"start": 0, "end": 1}]},
    # bad types
    {"a": [{"start": "1", "end": 2}]},
    {"a": [{"start": 1, "end": True}]},
    [{"start": 0, "end": 1}],
    "nope",
    42,
])
def test_api_invalid_unavailable_422(unavailable):
    resp = client.post("/solve", json=_payload(unavailable=unavailable))
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert "certainty" not in body


def test_api_old_request_regression():
    # Requests without the new field are byte-for-byte unchanged.
    payload = _payload(
        n=3, L=1, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 0, "b": 1}])
    default_resp = client.post("/solve", json=payload)
    assert default_resp.status_code == 200
    assert default_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "A"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    cont_resp = client.post(
        "/solve", json={**payload, "objective": "continuity"})
    assert cont_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "B"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]


def test_api_max_scale_with_windows():
    # Alternating position blocks force many short segments; the solver
    # must stay linear.
    import time

    n, L = 200_000, 4096
    rng = random.Random(313371)
    costs = [{"a": rng.randrange(0, 1_000_001),
              "b": rng.randrange(0, 1_000_001)} for _ in range(n)]
    win_a = [{"start": k, "end": k + 1} for k in range(1, n, 7)]
    win_b = [{"start": k, "end": k + 1} for k in range(4, n, 7)]
    t0 = time.perf_counter()
    resp = client.post("/solve", json={
        "n": n, "L": L, "fee_a": 10, "fee_b": 20, "costs": costs,
        "unavailable": {"a": win_a, "b": win_b}})
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
    assert resp.status_code == 200
    data = resp.json()
    segs = data["segments"]
    assert segs[0]["start"] == 0 and segs[-1]["end"] == n
    assert all(1 <= s["end"] - s["start"] <= L for s in segs)
    windows_by_source = {"A": (1, 7), "B": (4, 7)}
    blocked_by_source = {
        "A": {k for k in range(1, n, 7)},
        "B": {k for k in range(4, n, 7)},
    }
    for seg in segs:
        blocked = blocked_by_source[seg["source"]]
        for k in range(seg["start"], seg["end"]):
            assert k not in blocked
    assert len(data["certainty"]) == n
