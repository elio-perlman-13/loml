from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from wtv import Weapon, Target


@dataclass
class Solution:
    """
    Complete heuristic state: burst counts, fire times, weapon schedules,
    and target kill-chain. Supports deep copy for GRASP restarts and EIG.
    """
    # --- assignment decisions ---
    k: Dict[Tuple[int, int], int]                          # (weapon_id, target_id) -> burst count
    fire_times: Dict[Tuple[int, int], List[float]]         # (weapon_id, target_id) -> fire time list

    # --- mutable weapon state ---
    free: Dict[int, List[Tuple[float, float]]]             # weapon_id -> sorted free intervals
    remaining_ammo: Dict[int, int]                         # weapon_id -> remaining ammo after commits
    cap: Dict[Tuple[int, int], int]                        # (weapon_id, target_id) -> current capacity
    first_slot_cache: Dict[Tuple[int, int], Optional[float]] # (weapon_id, target_id) -> earliest feasible fire time

    # --- mutable target kill-chain state ---
    survival_rate: Dict[int, float]                        # target_id -> Π_i(1-p_ij)^k_ij
    threat_score: Dict[int, float]                         # target_id -> w_j (static)

    # --- static lookups (shared across copies, never mutated) ---
    p_ij: Dict[Tuple[int, int], float]                     # (weapon_id, target_id) -> p_kill per burst
    windows: Dict[Tuple[int, int], Tuple[float, float]]    # (weapon_id, target_id) -> (a_ij, b_ij)
    weapon_targets: Dict[int, List[int]]                   # weapon_id -> [target_ids] it can engage
    burst_duration: Dict[int, float]                       # weapon_id -> d_i = BurstInterval + ReloadTime
    max_shots: Dict[int, int]                              # weapon_id -> M_i (max bursts per target)
    vessel_id: Dict[int, int]                              # weapon_id -> vessel_id

    # ------------------------------------------------------------------ queries

    def survival(self, target_id: int) -> float:
        """w_j * Π_i (1-p_ij)^k_ij — current expected threat contribution."""
        return self.threat_score[target_id] * self.survival_rate[target_id]

    def objective(self) -> float:
        """Σ_j w_j Π_i (1-p_ij)^k_ij — lower is better."""
        return sum(self.survival(j) for j in self.threat_score)

    def count_slots(self, weapon_id: int, target_id: int) -> int:
        """Number of non-overlapping burst slots fitting in free[i] ∩ [a_ij, b_ij].

        Since free[i] is sorted and [a_ij, b_ij] is one contiguous window, binary
        search skips intervals entirely outside the window.  Within each overlapping
        free interval (s, e) the available length is min(e, b) - max(s, a), and the
        number of non-overlapping d-wide slots that fit is floor(available / d).
        """
        key = (weapon_id, target_id)
        if key not in self.windows:
            return 0
        a, b = self.windows[key]
        d = self.burst_duration[weapon_id]
        free = self.free[weapon_id]
        if not free or b - a < d:
            return 0

        # Binary search: first interval with start >= a; step back one to catch
        # intervals that started before a but end inside [a, b].
        idx = bisect.bisect_left(free, (a,))
        slots = 0
        for i in range(max(0, idx - 1), len(free)):
            s, e = free[i]
            if s >= b:           # all remaining intervals are past the window
                break
            length = min(e, b) - max(s, a)
            if length >= d:
                slots += int(length / d)
        return slots

    def _recompute_cap(self, weapon_id: int) -> None:
        """Recompute cap and first_slot_cache for all targets of weapon_id in a single pass.
        Called after every commit — ammo drop affects all j', free change affects both.
        Pairs whose capacity drops to zero are pruned from cap, first_slot_cache, and
        weapon_targets so the GRASP loop never visits them again.
        """
        ammo = self.remaining_ammo[weapon_id]
        M    = self.max_shots[weapon_id]
        d    = self.burst_duration[weapon_id]
        free = self.free[weapon_id]

        alive: List[int] = []
        for tid in self.weapon_targets.get(weapon_id, []):
            key  = (weapon_id, tid)
            a, b = self.windows[key]
            first: Optional[float] = None
            slots = 0

            if free and b - a >= d:
                idx = bisect.bisect_left(free, (a,))
                for i in range(max(0, idx - 1), len(free)):
                    s, e = free[i]
                    if s >= b:
                        break
                    length = min(e, b) - max(s, a)
                    if length >= d:
                        if first is None:
                            first = max(s, a)
                        slots += int(length / d)

            c = min(slots, ammo, M - self.k.get(key, 0))
            if c > 0:
                self.cap[key]              = c
                self.first_slot_cache[key] = first
                alive.append(tid)
            else:
                self.cap.pop(key, None)
                self.first_slot_cache.pop(key, None)

        self.weapon_targets[weapon_id] = alive

    def first_slot(self, weapon_id: int, target_id: int) -> Optional[float]:
        """Earliest feasible fire time — O(1) cache lookup; kept current by _recompute_cap."""
        return self.first_slot_cache.get((weapon_id, target_id))

    # ------------------------------------------------------------------ mutation

    def commit(self, weapon_id: int, target_id: int, t: float) -> None:
        """Commit one burst of weapon_id fired at time t against target_id."""
        key = (weapon_id, target_id)
        d = self.burst_duration[weapon_id]
        p = self.p_ij.get(key, 0.0)

        # burst count and fire time
        self.k[key] = self.k.get(key, 0) + 1
        self.fire_times.setdefault(key, []).append(t)

        # remove [t, t+d] from free[weapon_id]: bisect to the covering interval, splice in place
        free = self.free[weapon_id]
        end  = t + d
        idx  = bisect.bisect_right(free, (t, float("inf"))) - 1
        s, e = free[idx]
        replacements: List[Tuple[float, float]] = []
        if s < t:
            replacements.append((s, t))
        if end < e:
            replacements.append((end, e))
        free[idx : idx + 1] = replacements

        # ammo and kill-chain
        self.remaining_ammo[weapon_id] -= 1
        self.survival_rate[target_id] *= (1.0 - p)

        # maintain cap for all targets of this weapon
        self._recompute_cap(weapon_id)

    # ------------------------------------------------------------------ copy / factory

    def copy(self) -> Solution:
        """Deep copy of mutable state; static lookups are shared (read-only)."""
        return Solution(
            k=dict(self.k),
            fire_times={key: list(v) for key, v in self.fire_times.items()},
            free={wid: list(iv) for wid, iv in self.free.items()},
            remaining_ammo=dict(self.remaining_ammo),
            cap=dict(self.cap),
            first_slot_cache=dict(self.first_slot_cache),
            survival_rate=dict(self.survival_rate),
            threat_score=self.threat_score,
            p_ij=self.p_ij,
            windows=self.windows,
            weapon_targets=self.weapon_targets,
            burst_duration=self.burst_duration,
            max_shots=self.max_shots,
            vessel_id=self.vessel_id,
        )

    @classmethod
    def empty(
        cls,
        weapons: List[Weapon],
        targets: List[Target],
        p_ij: Dict[Tuple[int, int], float],
        windows: Dict[Tuple[int, int], Tuple[float, float]],
        horizon: float,
    ) -> Solution:
        """Initialize a blank solution from weapon/target instance lists."""
        ammo_map   = {w.id: w.ammo                for w in weapons}
        dur_map    = {w.id: w.burst_duration       for w in weapons}
        shots_map  = {w.id: w.max_shots_per_target for w in weapons}
        free_map   = {w.id: [(0.0, horizon)]       for w in weapons}

        # Prune statically infeasible pairs (window too narrow to fit one burst).
        weapon_targets: Dict[int, List[int]] = {}
        cap: Dict[Tuple[int, int], int] = {}
        first_slot_cache: Dict[Tuple[int, int], Optional[float]] = {}
        for (wid, tid), (a, b) in windows.items():
            d = dur_map[wid]
            if b - a < d:
                continue  # window too small — prune permanently
            c = min(int((b - a) / d), ammo_map[wid], shots_map[wid])
            if c <= 0:
                continue
            weapon_targets.setdefault(wid, []).append(tid)
            cap[(wid, tid)]              = c
            first_slot_cache[(wid, tid)] = a

        return cls(
            k={},
            fire_times={},
            free=free_map,
            remaining_ammo=ammo_map,
            cap=cap,
            first_slot_cache=first_slot_cache,
            survival_rate={t.id: 1.0 for t in targets},
            threat_score={t.id: t.threat_score for t in targets},
            p_ij=p_ij,
            windows=windows,
            weapon_targets=weapon_targets,
            burst_duration=dur_map,
            max_shots=shots_map,
            vessel_id={w.id: w.vessel_id for w in weapons},
        )

    # ------------------------------------------------------------------ output

    def assignments(self) -> List[dict]:
        """
        Return WTAAssignment-compatible records (one per committed weapon-target pair).
        PKill = 1 - (1-p_ij)^k_ij; FireTime = earliest fire time; EndTime = last burst end.
        """
        result = []
        for (wid, tid), count in self.k.items():
            if count == 0:
                continue
            p = self.p_ij.get((wid, tid), 0.0)
            times = self.fire_times.get((wid, tid), [])
            d = self.burst_duration[wid]
            result.append(dict(
                WTAVesselID=self.vessel_id[wid],
                WTAWeaponID=wid,
                WTATargetID=tid,
                AmmoUsed=count,
                PKill=round(1.0 - (1.0 - p) ** count, 6),
                FireTime=min(times) if times else 0.0,
                EndTime=max(times) + d if times else 0.0,
                FireTimes=sorted(times),
            ))
        return result