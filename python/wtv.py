from __future__ import annotations
import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class WeaponInfo:
    """Static weapon type properties (from weapon_infos lookup table)."""
    id: int
    code: str
    type: int
    min_range: float
    max_range: float
    min_altitude: float
    max_altitude: float
    azimuth_from_deg: float
    azimuth_to_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    max_shots_per_target: int   # M_i cap per target
    rounds_per_burst: int
    burst_interval: float       # BurstInterval — duration of one burst
    reload_time: float          # ReloadTime — gap after each burst

    @property
    def burst_duration(self) -> float:
        """d_i = BurstInterval + ReloadTime (total slot occupied per burst)."""
        return self.burst_interval + self.reload_time

    @classmethod
    def from_dict(cls, d: dict) -> WeaponInfo:
        return cls(
            id=d["ID"],
            code=d["Code"],
            type=d["Type"],
            min_range=d["MinRange"],
            max_range=d["MaxRange"],
            min_altitude=d["MinAltitude"],
            max_altitude=d["MaxAltitude"],
            azimuth_from_deg=d["AzimuthFromDeg"],
            azimuth_to_deg=d["AzimuthToDeg"],
            elevation_min_deg=d["ElevationMinDeg"],
            elevation_max_deg=d["ElevationMaxDeg"],
            max_shots_per_target=d["MaxShotsPerTarget"],
            rounds_per_burst=d["RoundsPerBurst"],
            burst_interval=d["BurstInterval"],
            reload_time=d["ReloadTime"],
        )


@dataclass
class TargetInfo:
    """Static target type properties (from target_infos lookup table)."""
    id: int
    code: str
    description: str
    type: int

    @classmethod
    def from_dict(cls, d: dict) -> TargetInfo:
        return cls(
            id=d["ID"],
            code=d["Code"],
            description=d.get("Description", ""),
            type=d["Type"],
        )


@dataclass
class Vessel:
    """A platform carrying weapons."""
    id: int
    x: float
    y: float
    z: float
    speed: float
    heading_x: float
    heading_y: float
    heading_z: float
    defense_radius: float

    @classmethod
    def from_dict(cls, d: dict) -> Vessel:
        return cls(
            id=d["ID"],
            x=d["X"],
            y=d["Y"],
            z=d["Z"],
            speed=d["Speed"],
            heading_x=d["HeadingX"],
            heading_y=d["HeadingY"],
            heading_z=d["HeadingZ"],
            defense_radius=d["DefenseRadius"],
        )


@dataclass
class Weapon:
    """A weapon instance mounted on a vessel."""
    id: int
    vessel_id: int
    ammo: int
    info_code: str
    status: int
    # --- heuristic state ---
    capable_bursts: int = 0                          # total bursts weapon can fire (ammo-limited)
    free: List[Tuple[float, float]] = field(         # free[i]: sorted non-overlapping free intervals (s, e)
        default_factory=list, repr=False
    )
    info: Optional[WeaponInfo] = field(default=None, repr=False)

    @property
    def burst_duration(self) -> float:
        """d_i = BurstInterval + ReloadTime — requires info to be resolved."""
        return self.info.burst_duration

    @property
    def max_shots_per_target(self) -> int:
        return self.info.max_shots_per_target

    def init_free(self, horizon: float) -> None:
        """Initialize free list to the full timeline [0, horizon]."""
        self.free = [(0.0, horizon)]

    def commit_interval(self, t: float) -> None:
        """Remove [t, t + burst_duration] from free, splitting the containing interval."""
        d = self.burst_duration
        new_free: List[Tuple[float, float]] = []
        for s, e in self.free:
            if e <= t or s >= t + d:
                new_free.append((s, e))          # no overlap
            else:
                if s < t:
                    new_free.append((s, t))      # left remnant
                if t + d < e:
                    new_free.append((t + d, e))  # right remnant
        self.free = new_free

    @classmethod
    def from_dict(cls, d: dict) -> Weapon:
        return cls(
            id=d["ID"],
            vessel_id=d["WTAVesselID"],
            ammo=d["Ammo"],
            info_code=d["WTAWeaponInfoCode"],
            status=d["Status"],
        )


@dataclass
class Target:
    """A threat target to be engaged."""
    id: int
    info_code: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    speed: float
    threat_score: float              # w_j — static threat weight
    # --- heuristic state ---
    survival_rate: float = 1.0       # Π_i (1 - p_ij)^k_ij, updated after each burst commit
    info: Optional[TargetInfo] = field(default=None, repr=False)

    @property
    def survival(self) -> float:
        """survival[j] = w_j * Π_i (1-p_ij)^k_ij — current expected threat contribution."""
        return self.threat_score * self.survival_rate

    def update_survival(self, p_ij: float) -> None:
        """Multiply survival_rate by (1 - p_ij) after committing one burst of weapon i."""
        self.survival_rate *= (1.0 - p_ij)

    @classmethod
    def from_dict(cls, d: dict) -> Target:
        return cls(
            id=d["ID"],
            info_code=d["WTATargetInfoCode"],
            x=d["X"],
            y=d["Y"],
            z=d["Z"],
            vx=d.get("VX", 0.0),
            vy=d.get("VY", 0.0),
            vz=d.get("VZ", 0.0),
            speed=d["Speed"],
            threat_score=d["ThreatScore"],
        )
