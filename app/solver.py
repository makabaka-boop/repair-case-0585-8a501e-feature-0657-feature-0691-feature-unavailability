"""Sliding-window dynamic programs for satellite telemetry gap repair.

A gap of ``n`` positions ``[0, n)`` is covered by back-to-back half-open
segments ``[start, end)``. Every segment uses source A or B. Choosing source
``s`` for ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the activation fee is paid once per segment, including when adjacent
segments use the same source.

Each source may carry a list of half-open maintenance windows over
positions. A source-``s`` segment is legal only when ``[i, j)`` does not
intersect any such window -- equivalently ``i`` and ``j`` lie in the same
availability block of source ``s``. The per-source windows are sorted,
non-overlapping and non-touching (validated by the API layer). Availability
is folded directly into every recurrence:

* For a forward DP ending a source-``s`` segment at ``j``, every legal
  predecessor ``i`` satisfies ``i >= floor_s[j]``, where ``floor_s[j]`` is
  the end of the last source-``s`` window ending before or at ``j`` (0 when
  the block contains 0). The window sweep therefore expires
  ``i < max(j-L, floor_s[j])``.
* For the backward suffix DP starting a source-``s`` segment at ``i``,
  every legal successor ``j`` satisfies ``j <= ceil_s[i]``, the start of
  the first window beginning strictly after ``i`` (``n`` in the last
  block). The reverse sweep expires ``j > min(i+L, ceil_s[i])``.

The bounds are two O(n) prefix/suffix sweeps over the windows; positions
and windows are never combined pairwise. When no cover can respect the
windows the cost-only prefix DP leaves ``f[n]`` infinite and the solver
raises ``NoFeasibleRepair`` before any plan or certainty is produced.

The default objective compares candidates lexicographically by
``(cost, segments, predecessor, source)``.

The ``continuity`` objective first keeps the minimum total cost. Among those
plans it minimizes source changes (the first segment is not a change), then
segment count, final-segment start, and final source. At each recurrence, a
tie in these values uses the selected prefix's own continuity ordering. It is
computed with separate A/B prefix states; the no-source sentinel is only used
for a first segment and is never inserted into an A/B state window.

Every successful solution also carries a per-position ``certainty`` label:
``A_ONLY`` / ``B_ONLY`` / ``EITHER``. Source ``s`` is "possible" at position
``k`` exactly when some globally minimum-cost legal cover contains a
source-``s`` segment covering ``k``. This set is objective-independent -- the
continuity switch count and the other secondary tie keys only pick one plan
out of the minimum-cost set and never narrow it -- so both objectives report
the same labels.

The certainty computation is a pure minimum-cost prefix/suffix split, both
of them obeying the same per-source availability bounds. A source-``s``
segment ``[i, j)`` can occur in a minimum-cost cover iff

    f[i] + fee[s] + (prefix_s[j] - prefix_s[i]) + g[j] == f[n]

where ``f[i]`` is the cheapest cost of covering ``[0, i)`` (over all sources)
and ``g[j]`` the cheapest cost of covering ``[j, n)`` (over all sources);
note the suffix does not depend on the source of the segment ending at ``j``.
For fixed ``(j, s)`` the minimizing prefixes ``i`` in the legal range
``[max(j-L, floor_s[j]), j-1]`` form a contiguous range; its leftmost
member is a sliding-window minimum of ``f[i] - prefix_s[i]`` and the
range is merged into a difference-array sweep. No segmentation is enumerated and winning-
predecessor replay is not enough, as it cannot reveal alternative optimal
predecessors.

Everything here uses O(n) time and O(n) space.
"""

from collections import deque
from dataclasses import dataclass, field

# Source ids double as tie-breakers: A < B.
SOURCE_A = 0
SOURCE_B = 1
SOURCE_NAMES = ("A", "B")

OBJECTIVE_DEFAULT = "default"
OBJECTIVE_CONTINUITY = "continuity"

_NO_SOURCE = -1

# Larger than any feasible total: at most n segments, each under
# fee(1e6) + L * 1e6 ~ 4.2e9, so n of them stays well below 1e18.
_INF = 10**30


class NoFeasibleRepair(Exception):
    """Raised when the maintenance windows make full coverage impossible."""

CERTAINTY_A_ONLY = "A_ONLY"
CERTAINTY_B_ONLY = "B_ONLY"
CERTAINTY_EITHER = "EITHER"


@dataclass(frozen=True)
class Segment:
    """A half-open segment [start, end) served by one source."""

    start: int
    end: int
    source: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "source": self.source}


@dataclass(frozen=True)
class Solution:
    cost: int
    segments: tuple[Segment, ...]
    certainty: tuple[str, ...] = field(default=())


def solve(
    n: int,
    costs: list[tuple[int, int]],
    fee_a: int,
    fee_b: int,
    max_len: int,
    objective: str = OBJECTIVE_DEFAULT,
    unavailable_a: tuple[tuple[int, int], ...] = (),
    unavailable_b: tuple[tuple[int, int], ...] = (),
) -> Solution:
    """Compute the optimal cover of [0, n) plus per-position certainty.

    ``costs[k]`` is the pair of per-position costs ``(cost of A, cost of
    B)`` at position ``k``. ``unavailable_a`` / ``unavailable_b`` list that
    source's half-open maintenance windows ``[start, end)``; windows are
    assumed already validated by the API layer (sorted, disjoint and
    non-touching, endpoints in ``[0, n]``).
    """
    fees = (fee_a, fee_b)
    prefixes = _prefix_sums(n, costs)
    floors, ceils = _availability_bounds(n, (unavailable_a, unavailable_b))

    # Feasibility is decided by the cost-only recurrence that also feeds
    # certainty; no plan is reconstructed for an infeasible instance.
    f = _min_prefix_costs(n, prefixes, fees, max_len, floors)
    if f[n] >= _INF:
        raise NoFeasibleRepair("maintenance windows prevent full coverage")

    if objective == OBJECTIVE_CONTINUITY:
        solution = _solve_continuity(
            n, prefixes, fees, max_len, floors)
    else:
        solution = _solve_default(n, prefixes, fees, max_len, floors)
    certainty = _certainty(
        n, prefixes, fees, max_len, f, floors, ceils)
    return Solution(solution.cost, solution.segments, tuple(certainty))


def _availability_bounds(
    n: int,
    unavailable: tuple[tuple[tuple[int, int], ...],
                      tuple[tuple[int, int], ...]],
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """Derive per-source availability-block boundary arrays.

    For a segment of source ``s`` ending at ``j``, every legal start ``i``
    satisfies ``i >= floors[s][j]``: this is 0 in the leading availability
    block, the end of the window after a closed window, and that window's
    own end while ``j`` lies inside it.

    For a segment of source ``s`` starting at ``i``, every legal end ``j``
    satisfies ``j <= ceils[s][i]``: this is ``n`` in the trailing block,
    the next window's start otherwise, and the window's own start while
    ``i`` lies inside it.

    Both arrays are linear sweeps over the disjoint, sorted windows -- each
    position index is written at most once -- so positions and windows are
    never crossed pairwise.
    """
    floors = ([0] * (n + 1), [0] * (n + 1))
    ceils = ([n] * (n + 1), [n] * (n + 1))
    for s, windows in enumerate(unavailable):
        if not windows:
            continue
        m = len(windows)
        floor_arr = floors[s]
        ceil_arr = ceils[s]

        # ---- floor_s[j] used by forward recurrences -------------------
        # Endpoints are partitioned into disjoint ranges across windows:
        #   [wstart+1, wend)          strict window interior (no segment
        #                             can end here: bound = wend expires
        #                             every predecessor i < j)
        #   [wend, next_start]        the following availability block
        #                             (next_start = next window's start,
        #                             or n for the last window)
        # j == wstart stays at the previous block's value (0 for the first
        # window): a segment may end exactly where a window starts.
        for t, (wstart, wend) in enumerate(windows):
            next_start = windows[t + 1][0] if t + 1 < m else n
            floor_arr[wstart + 1:wend] = [wend] * (wend - wstart - 1)
            floor_arr[wend:next_start + 1] = [wend] * (
                next_start - wend + 1)

        # ---- ceil_s[i] used by the backward recurrence ----------------
        # Mirror image with disjoint ranges:
        #   [0, first_start]          leading block
        #   [wstart+1, wend)          strict window interior
        #   [wend, next_start]        following block (next window's start)
        # i == wstart belongs to the preceding block; [last_end, n] stays n.
        first_start = windows[0][0]
        ceil_arr[0:first_start + 1] = [first_start] * (first_start + 1)
        for t, (wstart, wend) in enumerate(windows):
            ceil_arr[wstart + 1:wend] = [wstart] * (wend - wstart - 1)
            if t + 1 < m:
                next_start = windows[t + 1][0]
                ceil_arr[wend:next_start + 1] = [next_start] * (
                    next_start - wend + 1)
    return floors, ceils


def _prefix_sums(
    n: int, costs: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    prefix_a = [0] * (n + 1)
    prefix_b = [0] * (n + 1)
    for k in range(n):
        a, b = costs[k]
        prefix_a[k + 1] = prefix_a[k] + a
        prefix_b[k + 1] = prefix_b[k] + b
    return prefix_a, prefix_b


def _solve_default(n: int, prefixes: tuple[list[int], list[int]],
                   fees: tuple[int, int], max_len: int,
                   floors: tuple[list[int], list[int]]) -> Solution:
    # DP tables. Entries for unreachable endpoints keep cost _INF and are
    # never inserted into a predecessor window.
    best_cost = [_INF] * (n + 1)
    best_cost[0] = 0
    best_seg_count = [0] * (n + 1)
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    # Each entry in window s is an index i, keyed by
    # (best_cost[i] - prefix_s[i], best_seg_count[i], i).
    windows: tuple[deque, deque] = (deque([0]), deque([0]))

    for j in range(1, n + 1):
        best_candidate = None  # (cost, segments, prev, source)
        for s, window in enumerate(windows):
            low = max(j - max_len, floors[s][j])
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            candidate = (
                best_cost[i] - prefixes[s][i] + prefixes[s][j] + fees[s],
                best_seg_count[i] + 1,
                i,
                s,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        if best_candidate is None:
            continue

        cost, seg_count, i, s = best_candidate
        best_cost[j] = cost
        best_seg_count[j] = seg_count
        prev_index[j] = i
        prev_source[j] = s

        # Only reachable endpoints may serve as predecessors; unreachable
        # j's keep cost _INF and never enter a window.
        for s, window in enumerate(windows):
            key = (cost - prefixes[s][j], seg_count)
            while window:
                tail = window[-1]
                tail_key = (
                    best_cost[tail] - prefixes[s][tail],
                    best_seg_count[tail],
                )
                # Equal keys retain the smaller/older index; it has the
                # predecessor-index tie breaker and expires first.
                if tail_key <= key:
                    break
                window.pop()
            window.append(j)

    return _reconstruct(n, prev_index, prev_source, best_cost[n])


def _solve_continuity(n: int, prefixes: tuple[list[int], list[int]],
                      fees: tuple[int, int], max_len: int,
                      floors: tuple[list[int], list[int]]) -> Solution:
    # State p is the source of the last segment (0=A, 1=B). Arrays are
    # indexed [p][j]. The empty prefix is not state A or B. Unreachable
    # (p, j) pairs keep cost _INF and never enter a predecessor window.
    best_cost = [[_INF] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    seg_count = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    last_start = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    full_keys = [[()] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_index = [[-1] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_state = [[_NO_SOURCE] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]

    # windows[p][s] contains state-p prefixes to which a source-s segment is
    # appended. The two no-source rows hold only index 0 until it expires.
    windows = tuple(
        tuple(deque([0]) for _s in (SOURCE_A, SOURCE_B))
        for _p in range(3)
    )

    for j in range(1, n + 1):
        for s in (SOURCE_A, SOURCE_B):
            low = max(j - max_len, floors[s][j])
            best_candidate = None
            chosen_predecessor = _NO_SOURCE
            for p in range(3):
                window = windows[p][s]
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                if p < 2:
                    prefix_cost = best_cost[p][i]
                    prefix_switches = switches[p][i]
                    prefix_count = seg_count[p][i]
                    # The new final source is fixed, so a tie in the visible
                    # final-plan keys is resolved with the prefix's own full
                    # continuity order.
                    prefix_tie = full_keys[p][i]
                    predecessor_state = p
                    added_switch = 0 if p == s else 1
                else:
                    prefix_cost = 0
                    prefix_switches = 0
                    prefix_count = 0
                    prefix_tie = ()
                    predecessor_state = _NO_SOURCE
                    added_switch = 0

                candidate = (
                    prefix_cost - prefixes[s][i] + prefixes[s][j] + fees[s],
                    prefix_switches + added_switch,
                    prefix_count + 1,
                    i,
                    s,
                    prefix_tie,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    chosen_predecessor = predecessor_state

            if best_candidate is None:
                continue

            cost, switch_count, count, i, s, _prefix_tie = best_candidate
            best_cost[s][j] = cost
            switches[s][j] = switch_count
            seg_count[s][j] = count
            last_start[s][j] = i
            full_keys[s][j] = best_candidate
            prev_index[s][j] = i
            prev_state[s][j] = chosen_predecessor

            # A real A/B prefix becomes a predecessor only in windows keyed
            # by its actual last-source state.
            for new_s, window in enumerate(windows[s]):
                key = (
                    cost - prefixes[new_s][j],
                    switch_count,
                    count,
                )
                while window:
                    tail = window[-1]
                    tail_key = (
                        best_cost[s][tail] - prefixes[new_s][tail],
                        switches[s][tail],
                        seg_count[s][tail],
                    )
                    if tail_key <= key:
                        break
                    window.pop()
                window.append(j)

    final_candidate = None
    final_state = SOURCE_A
    for p in (SOURCE_A, SOURCE_B):
        candidate = (
            best_cost[p][n],
            switches[p][n],
            seg_count[p][n],
            last_start[p][n],
            p,
        )
        if final_candidate is None or candidate < final_candidate:
            final_candidate = candidate
            final_state = p

    return _reconstruct_states(
        n, prev_index, prev_state, final_state, final_candidate[0])


def _reconstruct(n: int, prev_index: list[int], prev_source: list[int],
                 total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    while end > 0:
        start = prev_index[end]
        segments.append(Segment(start, end, SOURCE_NAMES[prev_source[end]]))
        end = start
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


def _reconstruct_states(n: int, prev_index: list[list[int]],
                        prev_state: list[list[int]], final_state: int,
                        total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    state = final_state
    while end > 0:
        start = prev_index[state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[state][end]
        end = start
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


# --------------------------------------------------------------------------
# Per-position certainty over the set of globally minimum-cost covers
# --------------------------------------------------------------------------


def _min_prefix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int,
    floors: tuple[list[int], list[int]]
) -> list[int]:
    """f[j] = minimum cost of any legal cover of [0, j).

    Cost-only version of the default recurrence. Each per-source window is a
    monotone deque of indices ordered by f[i] - prefix_s[i]; expired indices
    leave at the front. Indices the availability blocks cannot reach stay
    infinite and are never enqueued.
    """
    f = [_INF] * (n + 1)
    f[0] = 0
    windows = (deque([0]), deque([0]))
    for j in range(1, n + 1):
        best = None
        for s, window in enumerate(windows):
            low = max(j - max_len, floors[s][j])
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            value = f[i] - prefixes[s][i] + prefixes[s][j] + fees[s]
            if best is None or value < best:
                best = value
        if best is None:
            continue
        f[j] = best
        for s, window in enumerate(windows):
            value = best - prefixes[s][j]
            while window:
                tail = window[-1]
                if f[tail] - prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return f


def _min_suffix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int,
    ceils: tuple[list[int], list[int]]
) -> list[int]:
    """g[j] = minimum cost of any legal cover of [j, n).

    The recurrence g[j] = min over (s, k) with j < k <= min(n, j+L) of
    fee[s] + prefix_s[k] - prefix_s[j] + g[k] rewrites, for a fixed
    successor k, as g[k] + fee[s] + prefix_s[k] minus prefix_s[j]. Indices k
    enter the window in descending order, so the monotone deques store
    indices in descending order and expiry (k > j + L, or k past the end of
    source s's block containing j) happens at the front.
    """
    g = [_INF] * (n + 1)
    g[n] = 0
    windows = (deque([n]), deque([n]))
    for j in range(n - 1, -1, -1):
        best = None
        for s, window in enumerate(windows):
            high = min(j + max_len, ceils[s][j])
            while window and window[0] > high:
                window.popleft()
            if not window:
                continue
            k = window[0]
            value = g[k] + fees[s] + prefixes[s][k] - prefixes[s][j]
            if best is None or value < best:
                best = value
        if best is None:
            continue
        g[j] = best
        for s, window in enumerate(windows):
            value = best + fees[s] + prefixes[s][j]
            while window:
                tail = window[-1]
                if g[tail] + fees[s] + prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return g


def _certainty(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int,
    f: list[int],
    floors: tuple[list[int], list[int]],
    ceils: tuple[list[int], list[int]],
) -> list[str]:
    """Label every position by which sources occur in minimum-cost covers.

    A source-s segment [i, j) participates in an optimal cover exactly when

        f[i] + fee[s] + P_s[j] - P_s[i] + g[j] == f[n]

    i.e. h_s(i) = f[i] - P_s[i] equals target = f[n] - g[j] - fee[s] -
    P_s[j]. Concatenating any prefix, the segment and any suffix is itself a
    complete feasible cover, so h_s(i) >= target for every feasible i (its
    cover costs at least the optimum); the segment is possible iff the
    window minimum equals target. Among ties the deque keeps the smallest
    predecessor, whose single segment [i_min, j) already covers the union of
    every minimizing predecessor's interval. Both the prefix window and the
    suffix feasibility honour the source's availability blocks.
    """
    g = _min_suffix_costs(n, prefixes, fees, max_len, ceils)
    optimum = f[n]

    # Per source: a monotone deque of predecessor indices ordered by
    # h_s(i) = f[i] - P_s[i]. Equal keys are retained (oldest first) so the
    # front is the smallest index attaining the window minimum.
    windows = (deque([0]), deque([0]))

    # difference[s] marks coverage of positions by source s over [0, n).
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(1, n + 1):
        if g[j] >= _INF:
            # No feasible suffix starts at j, so no optimal cover can end a
            # segment there; j is also unreachable and is not enqueued.
            continue
        for s, window in enumerate(windows):
            low = max(j - max_len, floors[s][j])
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            target = optimum - g[j] - fees[s] - prefixes[s][j]
            if f[i] - prefixes[s][i] == target:
                diff = difference[s]
                diff[i] += 1
                diff[j] -= 1

        if f[j] < _INF:
            for s, window in enumerate(windows):
                key = f[j] - prefixes[s][j]
                while window and f[window[-1]] - prefixes[s][window[-1]] > key:
                    window.pop()
                window.append(j)

    labels = [""] * n
    active = [0, 0]
    for k in range(n):
        active[SOURCE_A] += difference[SOURCE_A][k]
        active[SOURCE_B] += difference[SOURCE_B][k]
        a_possible = active[SOURCE_A] > 0
        b_possible = active[SOURCE_B] > 0
        if a_possible and b_possible:
            labels[k] = CERTAINTY_EITHER
        elif a_possible:
            labels[k] = CERTAINTY_A_ONLY
        elif b_possible:
            labels[k] = CERTAINTY_B_ONLY
        else:  # pragma: no cover - every position is covered by an optimum
            raise AssertionError("position uncovered by all optimal plans")
    return labels
