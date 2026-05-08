"""
heuristic.py — Phase-1 GRASP construction for the time-windowed multi-burst WTA problem.

Objective (minimise):  Σ_j  w_j · Π_i (1 - p_ij)^{k_ij}

Score of candidate (i, j, t):
    gain = survival[j] * p_ij
    opp  = Σ_{j' ≠ j, j' ∈ weapon_targets[i], cap[(i,j')] > 0}
               survival[j'] * ((1-p_{ij'})^{cap_after[(i,j')]} - (1-p_{ij'})^{cap[(i,j')]])
    score = gain + opp       (opp ≤ 0; penalises losing future capacity)

    cap_after is computed by simulating the free-list update and ammo decrement
    without committing, so scoring is side-effect free.
"""

from __future__ import annotations

import bisect
import random
from typing import Dict, List, Optional, Tuple

from solution import Solution
from wtv import Weapon, Target


# ---------------------------------------------------------------------------
# Opportunity-cost helpers
# ---------------------------------------------------------------------------

def _free_after_commit(
    free: List[Tuple[float, float]],
    t: float,
    d: float,
) -> List[Tuple[float, float]]:
    """Return a copy of sol.free[wid] (sorted list of disjoint intervals) with [t, t+d] removed.

    Uses bisect to find the unique covering interval in O(log n), then
    splices the two sub-intervals in place on the copy — O(1) instead of a full rebuild.
    """
    # Last interval with start <= t: bisect_right with sentinel (t, inf) steps over any (t, e).
    idx = bisect.bisect_right(free, (t, float("inf"))) - 1
    result = list(free)  # shallow copy — tuples are immutable
    s, e = result[idx]
    end = t + d
    replacements: List[Tuple[float, float]] = []
    if s + d <= t: # only append when the remnant interval is large enough to host another burst
        replacements.append((s, t))
    if end + d <= e:
        replacements.append((end, e))
    result[idx : idx + 1] = replacements  # sorted order preserved by construction
    return result


def _count_slots_with_free(
    free: List[Tuple[float, float]],
    a: float,
    b: float,
    d: float,
) -> int:
    """count_slots against an arbitrary free list (not bound to a Solution instance)."""
    if not free or b - a < d:
        return 0
    idx = bisect.bisect_left(free, (a,))
    slots = 0
    for i in range(max(0, idx - 1), len(free)):
        s, e = free[i]
        if s >= b:
            break
        length = min(e, b) - max(s, a)
        if length >= d:
            slots += int(length / d)
    return slots


def _opp_cost(sol: Solution, wid: int, tid: int, t: float) -> float:
    """
    Opportunity cost (≤ 0) of committing burst (wid, tid) fired at time t.

    For every other target j' that wid can still engage (cap > 0, j' ≠ tid):
        cap_after[(wid, j')] = min(sched_new, ammo-1, M - k[(wid,j')])
        Δ = survival[j'] * ((1-p_{ij'})^{cap_after} - (1-p_{ij'})^{cap_before})  ≤ 0

    Returns Σ Δ — the total reduction in future kill potential caused by this commit.
    """
    d        = sol.burst_duration[wid]
    ammo_new = sol.remaining_ammo[wid] - 1
    M        = sol.max_shots[wid]
    free_new = _free_after_commit(sol.free[wid], t, d)

    opp = 0.0
    for jp in sol.weapon_targets.get(wid, []):
        if jp == tid:
            continue
        key = (wid, jp)
        cap_before = sol.cap.get(key, 0)
        if cap_before <= 0:
            continue  # already saturated or unreachable — no loss

        a, b = sol.windows[key]
        sched_new = _count_slots_with_free(free_new, a, b, d)
        k_jp      = sol.k.get(key, 0)
        cap_after = max(min(sched_new, ammo_new, M - k_jp), 0)

        if cap_after == cap_before:
            continue  # no capacity loss — skip

        p       = sol.p_ij.get(key, 0.0)
        surv_jp = sol.survival(jp)
        # (1-p)^cap_after - (1-p)^cap_before ≤ 0 when cap_after < cap_before
        opp += surv_jp * ((1.0 - p) ** cap_after - (1.0 - p) ** cap_before)

    return opp  # ≤ 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(sol: Solution, wid: int, tid: int, t: float) -> float:
    """
    Approximate rollout score for committing (wid, tid, t).

        score = gain + opp
              = survival[j] * p_ij
                + Σ_{j'≠j} survival[j'] * ((1-p_{ij'})^{cap_after} - (1-p_{ij'})^{cap_before})

    Higher score → better assignment.
    """
    p    = sol.p_ij.get((wid, tid), 0.0)
    threat =sol.threat_score.get(tid, 0.0)
    gain = sol.survival(tid) * p * threat
   opp  = _opp_cost(sol, wid, tid, t)
    return gain


# ---------------------------------------------------------------------------
# Phase 1: GRASP construction
# ---------------------------------------------------------------------------

def grasp_construction(sol: Solution, alpha: float, rng: random.Random) -> Solution:
    """
    Greedy Randomised Adaptive Search construction phase.

    At each iteration:
      1. Build the candidate list of all feasible (wid, tid) pairs with cap > 0.
      2. Score each candidate with the approximate rollout criterion.
      3. Restrict to the Restricted Candidate List (RCL):
             score ≥ (1 - alpha) * best_score
      4. Uniformly draw one candidate from the RCL and commit it.

    Terminates when no feasible candidate with a positive score remains.

    Parameters
    ----------
    sol   : A blank (or partially filled) Solution — mutated in place.
    alpha : Greediness ∈ [0, 1].  0 = pure greedy;  1 = fully random feasible.
    rng   : Random source for reproducible stochastic selection.
    """
    while True:
        candidates: List[Tuple[Tuple[int, int, float], float]] = []

        for (wid, tid), c in sol.cap.items():
            if c <= 0:
                continue
            t = sol.first_slot(wid, tid)
            if t is None:
                continue
            s = _score(sol, wid, tid, t)
            candidates.append(((wid, tid, t), s))

        if not candidates:
            break

        best = max(s for _, s in candidates)
        if best <= 0.0:
            # Every remaining candidate would increase (or not reduce) expected threat.
            break

        threshold = (1.0 - alpha) * best
        rcl = [(a, s) for a, s in candidates if s >= threshold]

        (wid, tid, t), _ = rng.choice(rcl)
        sol.commit(wid, tid, t)

    return sol


def grasp(
    weapons: List[Weapon],
    targets: List[Target],
    p_ij: Dict[Tuple[int, int], float],
    windows: Dict[Tuple[int, int], Tuple[float, float]],
    horizon: float,
    *,
    alpha: float = 0.15,
    restarts: int = 10,
    seed: Optional[int] = None,
) -> Solution:
    """
    Run `restarts` independent GRASP constructions and return the best solution.

    Parameters
    ----------
    weapons  : List of Weapon instances.
    targets  : List of Target instances.
    p_ij     : Kill-probability per burst, keyed (weapon_id, target_id).
    windows  : Engagement windows, keyed (weapon_id, target_id) → (a_ij, b_ij).
    horizon  : End of the planning horizon (weapons are free in [0, horizon] initially).
    alpha    : RCL greediness ∈ [0, 1].
    restarts : Number of independent GRASP restarts.
    seed     : RNG seed for reproducibility (None → non-deterministic).

    Returns
    -------
    The Solution with the lowest objective value across all restarts.
    """
    rng  = random.Random(seed)
    best: Optional[Solution] = None

    for _ in range(restarts):
        sol = Solution.empty(weapons, targets, p_ij, windows, horizon)
        sol = grasp_construction(sol, alpha, rng)
        if best is None or sol.objective() < best.objective():
            best = sol

    assert best is not None, "restarts must be >= 1"
    return best
