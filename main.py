#!/usr/bin/env python3
"""
main.py — Load a WTA scenario JSON and run GRASP to produce a collection of solutions.

Usage:
    python main.py [scenario.json] [--restarts N] [--alpha A] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from wtv import Weapon, WeaponInfo, Target, TargetInfo
from solution import Solution
from heuristic import grasp


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> Tuple[
    List[Weapon],
    List[Target],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], Tuple[float, float]],
    float,
]:
    """Parse scenario JSON and return (weapons, targets, p_ij, windows, horizon)."""
    with open(path) as f:
        data = json.load(f)

    # --- static catalogs ---
    weapon_info_by_code = {d["Code"]: WeaponInfo.from_dict(d) for d in data["weapon_infos"]}
    target_info_by_code = {d["Code"]: TargetInfo.from_dict(d) for d in data["target_infos"]}

    # p_kill per burst: (weapon_info_code, target_info_code) -> float
    prob_by_codes: Dict[Tuple[str, str], float] = {}
    for row in data["probability_table"]:
        prob_by_codes[(row["WTAWeaponInfoCode"], row["WTATargetInfoCode"])] = row["Score"]

    req = data["assignment_request"]

    # --- weapons ---
    weapons: List[Weapon] = []
    for d in req["weapons"]:
        w = Weapon.from_dict(d)
        w.info = weapon_info_by_code[w.info_code]
        weapons.append(w)
    weapon_by_id = {w.id: w for w in weapons}

    # --- targets ---
    targets: List[Target] = []
    for d in req["targets"]:
        t = Target.from_dict(d)
        t.info = target_info_by_code[t.info_code]
        targets.append(t)
    target_by_id = {t.id: t for t in targets}

    # --- engagement windows and p_ij ---
    windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    p_ij: Dict[Tuple[int, int], float] = {}

    for key_str, (a, b) in data["engagement_windows"].items():
        wid, tid = map(int, key_str.split("_"))
        w = weapon_by_id[wid]
        t = target_by_id[tid]
        p = prob_by_codes.get((w.info_code, t.info_code), 0.0)
        if p <= 0.0:
            continue  # no kill probability — skip pair entirely
        windows[(wid, tid)] = (a, b)
        p_ij[(wid, tid)]    = p

    horizon = max(b for _, b in windows.values()) if windows else 60.0

    return weapons, targets, p_ij, windows, horizon


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_solution(sol: Solution, rank: int) -> None:
    obj = sol.objective()
    assignments = sol.assignments()
    print(f"\n{'='*60}")
    print(f"Solution #{rank}  objective={obj:.6f}  assignments={len(assignments)}")
    print(f"{'='*60}")
    for a in sorted(assignments, key=lambda x: (x["WTAVesselID"], x["WTAWeaponID"], x["WTATargetID"])):
        print(
            f"  vessel={a['WTAVesselID']}  weapon={a['WTAWeaponID']:>4}  "
            f"target={a['WTATargetID']:>4}  bursts={a['AmmoUsed']}  "
            f"pkill={a['PKill']:.4f}  fire={a['FireTime']:.2f}s  end={a['EndTime']:.2f}s"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WTA GRASP solver")
    parser.add_argument("scenario", nargs="?",
                        default="/workspaces/WTA/data/scenario_001.json",
                        help="Path to scenario JSON (default: scenario_001.json)")
    parser.add_argument("--restarts", type=int, default=10,
                        help="Number of GRASP restarts (default: 10)")
    parser.add_argument("--alpha",    type=float, default=0.15,
                        help="RCL greediness α ∈ [0,1] (default: 0.15)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="RNG seed (default: 42)")
    args = parser.parse_args()

    path = Path(args.scenario)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {path} ...")
    weapons, targets, p_ij, windows, horizon = load_scenario(str(path))
    print(f"  weapons={len(weapons)}  targets={len(targets)}  "
          f"pairs={len(windows)}  horizon={horizon:.1f}s")

    print(f"\nRunning GRASP: restarts={args.restarts}  alpha={args.alpha}  seed={args.seed}")
    t0 = time.perf_counter()

    # Collect one Solution per restart (grasp() internally keeps only the best,
    # so run individual restarts to get the full collection).
    import random
    rng = random.Random(args.seed)
    from heuristic import grasp_construction

    solutions: List[Solution] = []
    for r in range(args.restarts):
        sol = Solution.empty(weapons, targets, p_ij, windows, horizon)
        sol = grasp_construction(sol, args.alpha, rng)
        solutions.append(sol)
        print(f"  restart {r+1:>3}/{args.restarts}  obj={sol.objective():.6f}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.2f}s")

    # Sort by objective (best first) and display
    solutions.sort(key=lambda s: s.objective())
    """ for rank, sol in enumerate(solutions, 1):
        print_solution(sol, rank) """

    best = solutions[0]
    print(f"\nBest objective: {best.objective():.6f}")


if __name__ == "__main__":
    main()
