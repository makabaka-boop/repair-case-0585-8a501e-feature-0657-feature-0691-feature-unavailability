"""Sliding-window dynamic programs for satellite telemetry gap repair.

A gap of ``n`` positions ``[0, n)`` is covered by back-to-back half-open
segments ``[start, end)``. Every segment uses source A or B. Choosing source
``s`` for ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the activation fee is paid once per segment, including when adjacent
segments use the same source.

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

The certainty computation is a pure minimum-cost prefix/suffix split. A
source-``s`` segment ``[i, j)`` can occur in a minimum-cost cover iff

    f[i] + fee[s] + (prefix_s[j] - prefix_s[i]) + g[j] == f[n]

where ``f[i]`` is the cheapest cost of covering ``[0, i)`` (over all sources)
and ``g[j]`` the cheapest cost of covering ``[j, n)`` (over all sources);
note the suffix does not depend on the source of the segment ending at ``j``.
For fixed ``(j, s)`` the minimizing prefixes ``i`` in the legal window
``[j-L, j-1]`` form a contiguous range; its leftmost member is a
sliding-window minimum of ``f[i] - prefix_s[i]`` and the range is merged into
a difference-array sweep. No segmentation is enumerated and winning-
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


def solve(n: int, costs: list[tuple[int, int]], fee_a: int, fee_b: int,
          max_len: int, objective: str = OBJECTIVE_DEFAULT) -> Solution:
    """Compute the optimal cover of [0, n) plus per-position certainty.

    ``costs[k]`` is the pair of per-position costs ``(cost of A, cost of
    B)`` at position ``k``. Inputs are assumed already validated by the API
    layer; the algorithm itself trusts the bounds.
    """
    fees = (fee_a, fee_b)
    prefixes = _prefix_sums(n, costs)
    if objective == OBJECTIVE_CONTINUITY:
        solution = _solve_continuity(n, prefixes, fees, max_len)
    else:
        solution = _solve_default(n, prefixes, fees, max_len)
    certainty = _certainty(n, prefixes, fees, max_len)
    return Solution(solution.cost, solution.segments, tuple(certainty))


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
                   fees: tuple[int, int], max_len: int) -> Solution:
    # DP tables.
    best_cost = [0] * (n + 1)
    best_seg_count = [0] * (n + 1)
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    # Each entry in window s is an index i, keyed by
    # (best_cost[i] - prefix_s[i], best_seg_count[i], i).
    windows: tuple[deque, deque] = (deque([0]), deque([0]))

    for j in range(1, n + 1):
        low = j - max_len
        for window in windows:
            while window and window[0] < low:
                window.popleft()

        best_candidate = None  # (cost, segments, prev, source)
        for s, window in enumerate(windows):
            i = window[0]
            candidate = (
                best_cost[i] - prefixes[s][i] + prefixes[s][j] + fees[s],
                best_seg_count[i] + 1,
                i,
                s,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        cost, seg_count, i, s = best_candidate
        best_cost[j] = cost
        best_seg_count[j] = seg_count
        prev_index[j] = i
        prev_source[j] = s

        for s, window in enumerate(windows):
            key = (best_cost[j] - prefixes[s][j], best_seg_count[j])
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
                      fees: tuple[int, int], max_len: int) -> Solution:
    # State p is the source of the last segment (0=A, 1=B). Arrays are
    # indexed [p][j]. The empty prefix is not state A or B.
    best_cost = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
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
        low = j - max_len
        for row in windows:
            for window in row:
                while window and window[0] < low:
                    window.popleft()

        for s in (SOURCE_A, SOURCE_B):
            best_candidate = None
            chosen_predecessor = _NO_SOURCE
            for p in range(3):
                window = windows[p][s]
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
                    best_cost[s][j] - prefixes[new_s][j],
                    switches[s][j],
                    seg_count[s][j],
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
    fees: tuple[int, int], max_len: int
) -> list[int]:
    """f[j] = minimum cost of any legal cover of [0, j).

    Cost-only version of the default recurrence. Each per-source window is a
    monotone deque of indices ordered by f[i] - prefix_s[i]; expired indices
    leave at the front.
    """
    f = [0] * (n + 1)
    windows = (deque([0]), deque([0]))
    for j in range(1, n + 1):
        low = j - max_len
        best = None
        for s, window in enumerate(windows):
            while window and window[0] < low:
                window.popleft()
            i = window[0]
            value = f[i] - prefixes[s][i] + prefixes[s][j] + fees[s]
            if best is None or value < best:
                best = value
        f[j] = best
        for s, window in enumerate(windows):
            value = f[j] - prefixes[s][j]
            while window:
                tail = window[-1]
                if f[tail] - prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return f


def _min_suffix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int
) -> list[int]:
    """g[j] = minimum cost of any legal cover of [j, n).

    The recurrence g[j] = min over (s, k) with j < k <= min(n, j+L) of
    fee[s] + prefix_s[k] - prefix_s[j] + g[k] rewrites, for a fixed
    successor k, as g[k] + fee[s] + prefix_s[k] minus prefix_s[j]. Indices k
    enter the window in descending order, so the monotone deques store
    indices in descending order and expiry (k > j + L) happens at the front.
    """
    g = [0] * (n + 1)
    windows = (deque([n]), deque([n]))
    for j in range(n - 1, -1, -1):
        high = j + max_len
        best = None
        for s, window in enumerate(windows):
            while window and window[0] > high:
                window.popleft()
            k = window[0]
            value = g[k] + fees[s] + prefixes[s][k] - prefixes[s][j]
            if best is None or value < best:
                best = value
        g[j] = best
        for s, window in enumerate(windows):
            value = g[j] + fees[s] + prefixes[s][j]
            while window:
                tail = window[-1]
                if g[tail] + fees[s] + prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return g


def _certainty(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int
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
    every minimizing predecessor's interval.
    """
    f = _min_prefix_costs(n, prefixes, fees, max_len)
    g = _min_suffix_costs(n, prefixes, fees, max_len)
    optimum = f[n]

    # Per source: a monotone deque of predecessor indices ordered by
    # h_s(i) = f[i] - P_s[i]. Equal keys are retained (oldest first) so the
    # front is the smallest index attaining the window minimum.
    windows = (deque([0]), deque([0]))

    # difference[s] marks coverage of positions by source s over [0, n).
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(1, n + 1):
        low = j - max_len
        if low < 0:
            low = 0
        for s, window in enumerate(windows):
            while window and window[0] < low:
                window.popleft()
            i = window[0]
            target = optimum - g[j] - fees[s] - prefixes[s][j]
            if f[i] - prefixes[s][i] == target:
                diff = difference[s]
                diff[i] += 1
                diff[j] -= 1

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
