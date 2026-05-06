#!/usr/bin/env python3
"""
WTA Scenario Data Generator
Outputs JSON matching CMS_WTA.proto schema (WTAAssignmentRequest + catalogs).

Coordinate system: flat-earth Cartesian, km, relative to vessel 1.
  X = East, Y = North, Z = Up
Speeds in km/s.  Heading vectors are unit 3-D vectors (X, Y, Z).
"""

import json
import math
import random
import argparse

random.seed(42)

# ── Weapon ranges are in metres in the catalogue; internally we work in km ─────
M2KM = 1e-3
KM2M = 1e3

# ── Weapon Info Catalog ────────────────────────────────────────────────────────
#  Type 1 = SAM (Tên lửa phòng không)
#  Type 2 = Cannon/CIWS (Pháo)
#  Type 3 = EW/Jamming (TCĐT)
#  Type 4 = Anti-ship missile (Tên lửa VCM)
WEAPON_INFOS = [
    # ── Tên lửa phòng không – 360°, targets air threats only, range 1.5–10 km
    # Per ship: 2 giàn × 4 ống/giàn = 8 rounds total
    dict(ID=1, Code="SAM_PK", Type=1,
         MinRange=1_500,  MaxRange=10_000, MinAltitude=10,  MaxAltitude=8_000,
         AzimuthFromDeg=0, AzimuthToDeg=360,
         ElevationMinDeg=2,  ElevationMaxDeg=85,
         MaxShotsPerTarget=4, RoundsPerBurst=1, BurstInterval=3.0, ReloadTime=6.0),

    # ── Pháo AK176 – single mount fore, 12 km, ±175° (covers almost all but tight aft)
    # 252 rounds/unit; fires bursts of ~10 rds
    dict(ID=2, Code="AK176", Type=2,
         MinRange=500,    MaxRange=12_000, MinAltitude=-5,  MaxAltitude=3_000,
         AzimuthFromDeg=5, AzimuthToDeg=355,   # ±175° from bow
         ElevationMinDeg=-5,  ElevationMaxDeg=60,
         MaxShotsPerTarget=15, RoundsPerBurst=10, BurstInterval=1.2, ReloadTime=1.0),

    # ── Pháo AK630M – 2 mounts (port & starboard), 4 km
    # Góc ngang: 30°–115° each side (port: 30–115, stbd: 245–330 from North relative to ship)
    # 3000 rounds/unit; very high ROF
    dict(ID=3, Code="AK630_PORT", Type=2,
         MinRange=100,    MaxRange=4_000,  MinAltitude=-10, MaxAltitude=2_000,
         AzimuthFromDeg=30,  AzimuthToDeg=115,
         ElevationMinDeg=-5,  ElevationMaxDeg=70,
         MaxShotsPerTarget=50, RoundsPerBurst=100, BurstInterval=0.4, ReloadTime=0.3),
    dict(ID=4, Code="AK630_STBD", Type=2,
         MinRange=100,    MaxRange=4_000,  MinAltitude=-10, MaxAltitude=2_000,
         AzimuthFromDeg=245, AzimuthToDeg=330,
         ElevationMinDeg=-5,  ElevationMaxDeg=70,
         MaxShotsPerTarget=50, RoundsPerBurst=100, BurstInterval=0.4, ReloadTime=0.3),

    # ── TCĐT BV15 – jamming system, 1 tổ hợp
    # Effective range: UAV 10 km, ASM 20 km; 360° coverage
    dict(ID=5, Code="TCDT_BV15", Type=3,
         MinRange=0,      MaxRange=20_000, MinAltitude=0,   MaxAltitude=8_000,
         AzimuthFromDeg=0, AzimuthToDeg=360,
         ElevationMinDeg=0,  ElevationMaxDeg=90,
         MaxShotsPerTarget=6, RoundsPerBurst=1, BurstInterval=2.0, ReloadTime=12.0),

    # ── Tên lửa VCM – anti-ship only, 10–300 km, 360°
    # 2 bệ × 4 ống/bệ = 8 rounds
    dict(ID=6, Code="VCM", Type=4,
         MinRange=10_000, MaxRange=300_000, MinAltitude=-5, MaxAltitude=50,
         AzimuthFromDeg=0, AzimuthToDeg=360,
         ElevationMinDeg=-2, ElevationMaxDeg=10,
         MaxShotsPerTarget=2, RoundsPerBurst=1, BurstInterval=8.0, ReloadTime=30.0),
]
WI = {w["Code"]: w for w in WEAPON_INFOS}

# ── Target Info Catalog ────────────────────────────────────────────────────────
# Types: 1=ASM, 2=Aircraft, 3=UAV, 4=Surface vessel, 5=USV, 6=Torpedo
TARGET_INFOS = [
    dict(ID=1, Code="TGT_ASM",        Description="Anti-Ship Missile (ASCM)", Type=1),
    dict(ID=2, Code="TGT_FIGHTER",    Description="Fighter Jet",              Type=2),
    dict(ID=3, Code="TGT_HELICOPTER", Description="Helicopter",               Type=2),
    dict(ID=4, Code="TGT_UAV_FIXED",  Description="Fixed-wing UAV",           Type=3),
    dict(ID=5, Code="TGT_UAV_ROTOR",  Description="Rotary UAV",               Type=3),
    dict(ID=6, Code="TGT_KAMIKAZE",   Description="Kamikaze Drone",           Type=3),
    dict(ID=7, Code="TGT_SURFACE",    Description="Surface Warship",          Type=4),
    dict(ID=8, Code="TGT_USV",        Description="Unmanned Surface Vehicle",  Type=5),
    dict(ID=9, Code="TGT_TORPEDO",    Description="Torpedo",                  Type=6),
]

# ── Probability Table ──────────────────────────────────────────────────────────
# Source: p_kill matrix from ship spec (0 = weapon cannot engage that target class)
#
#                   ASM   FIGHT  HELI   UAV_F  UAV_R  KAMI   SURF   USV    TORP
_TGT_CODES = [t["Code"] for t in TARGET_INFOS]
_PROB_MATRIX = {
    # Tên lửa phòng không – air threats only
    "SAM_PK":      [0.40, 0.70,  0.70,  0.75,  0.75,  0.75,  0.00,  0.00,  0.00],
    # Pháo AK176 – surface & slow air; not effective vs fast ASM
    "AK176":       [0.00, 0.00,  0.00,  0.00,  0.00,  0.00,  0.30,  0.60,  0.60],
    # Pháo AK630M (port & stbd same p_kill) – close-in; ASM + air + surface
    "AK630_PORT":  [0.20, 0.50,  0.50,  0.60,  0.60,  0.60,  0.20,  0.40,  0.40],
    "AK630_STBD":  [0.20, 0.50,  0.50,  0.60,  0.60,  0.60,  0.20,  0.40,  0.40],
    # TCĐT BV15 – jamming: effective vs ASM & UAV/drone; not kinetic vs surface
    "TCDT_BV15":   [0.50, 0.00,  0.00,  0.80,  0.80,  0.80,  0.00,  0.80,  0.00],
    # VCM – anti-ship missile: surface targets only
    "VCM":         [0.00, 0.00,  0.00,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00],
}
PROB_TABLE = [
    dict(Score=score, WTAWeaponInfoCode=wcode, WTATargetInfoCode=tcode)
    for wcode, scores in _PROB_MATRIX.items()
    for tcode, score in zip(_TGT_CODES, scores)
    if score > 0.0   # only store non-zero entries
]

# ── Target physical properties ─────────────────────────────────────────────────
# primary: weapons used to place the target in a valid envelope during generation
TARGET_PROPS = {
    # ASM: sea-skimming 3–15 m, Mach 0.85–0.9 (~290–310 m/s)
    "TGT_ASM":        dict(speed=(270, 310), alt=(3,   15),   threat=(0.85, 1.00), pitch=(-0.5,  0.5),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD", "SAM_PK"]),
    # Fighter: high-alt, fast
    "TGT_FIGHTER":    dict(speed=(200, 320), alt=(500, 6000), threat=(0.65, 0.90), pitch=(-30.0, 20.0),
                           primary=["SAM_PK"]),
    # Helicopter: slow, low
    "TGT_HELICOPTER": dict(speed=(25,  70),  alt=(20,  400),  threat=(0.35, 0.65), pitch=(-10.0, 10.0),
                           primary=["SAM_PK", "AK630_PORT", "AK630_STBD"]),
    # Fixed-wing UAV
    "TGT_UAV_FIXED":  dict(speed=(30,  80),  alt=(100, 800),  threat=(0.45, 0.75), pitch=(-5.0,  5.0),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Rotary UAV: very slow, low
    "TGT_UAV_ROTOR":  dict(speed=(5,   30),  alt=(10,  200),  threat=(0.40, 0.70), pitch=(-8.0,  8.0),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Kamikaze drone: medium speed, low-mid alt, dives at terminal phase
    "TGT_KAMIKAZE":   dict(speed=(50, 130),  alt=(20,  500),  threat=(0.55, 0.85), pitch=(-15.0, 5.0),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Surface warship: slow, at sea level
    "TGT_SURFACE":    dict(speed=(8,   20),  alt=(0,   5),    threat=(0.60, 0.95), pitch=(0.0,  0.0),
                           primary=["VCM", "AK176"]),
    # USV: small, fast unmanned surface vessel
    "TGT_USV":        dict(speed=(15,  35),  alt=(0,   3),    threat=(0.40, 0.70), pitch=(0.0,  0.0),
                           primary=["AK176", "AK630_PORT", "AK630_STBD"]),
    # Torpedo: very fast underwater – treat altitude as slightly negative
    "TGT_TORPEDO":    dict(speed=(20,  40),  alt=(-3,  0),    threat=(0.50, 0.80), pitch=(0.0,  0.0),
                           primary=["AK176", "AK630_PORT", "AK630_STBD"]),
}

# Number of targets per type (sums to 100)
# ── Min per type (floors for random distribution) ────────────────────────────
_TGT_MIN = {
    "TGT_ASM": 8, "TGT_FIGHTER": 5, "TGT_HELICOPTER": 5,
    "TGT_UAV_FIXED": 8, "TGT_UAV_ROTOR": 8, "TGT_KAMIKAZE": 8,
    "TGT_SURFACE": 4, "TGT_USV": 5, "TGT_TORPEDO": 5,
}
_WPN_MIN = {
    "SAM_PK": 2, "AK176": 1, "AK630_PORT": 2, "AK630_STBD": 2,
    "TCDT_BV15": 1, "VCM": 1,
}
# Ammo: (lo_min, lo_max, hi_min, hi_max)
_AMMO_BOUNDS = {
    "SAM_PK":      (4,  6,   8, 10),   # 2 giàn × 4 ống = 8 base
    "AK176":       (150, 200, 220, 252), # 252 rounds/unit
    "AK630_PORT":  (2000, 2500, 2800, 3000),
    "AK630_STBD":  (2000, 2500, 2800, 3000),
    "TCDT_BV15":   (6,  8,  10, 15),   # jammer activations
    "VCM":         (4,  6,   7,  8),   # 2 bệ × 4 ống
}

def _random_partition(total, keys, min_per_key: dict):
    """Randomly distribute `total` integer units across `keys`,
    each key getting at least min_per_key[key] units."""
    floor_total = sum(min_per_key[k] for k in keys)
    assert floor_total <= total, "floors exceed total"
    remainder = total - floor_total
    # random breakpoints in [0, remainder] (with repetition → some slices can be 0)
    n = len(keys)
    pts = sorted([random.randint(0, remainder) for _ in range(n - 1)])
    pts = [0] + pts + [remainder]
    shares = [pts[i + 1] - pts[i] + min_per_key[k] for i, k in enumerate(keys)]
    # shuffle so the extra units aren't always front-loaded
    extra = [s - min_per_key[k] for s, k in zip(shares, keys)]
    random.shuffle(extra)
    return {k: min_per_key[k] + extra[i] for i, k in enumerate(keys)}

def randomize_scenario_params(n_targets: int = 100, n_weapons_per_vessel: int = 50):
    """Return (target_counts, weapon_dist, ammo_range) sampled randomly."""
    tgt_keys = list(_TGT_MIN.keys())
    wpn_keys = list(_WPN_MIN.keys())

    target_counts  = _random_partition(n_targets, tgt_keys, _TGT_MIN)
    weapon_dist    = _random_partition(n_weapons_per_vessel, wpn_keys, _WPN_MIN)

    ammo_range = {}
    for k, (lo_min, lo_max, hi_min, hi_max) in _AMMO_BOUNDS.items():
        lo = random.randint(lo_min, lo_max)
        hi = random.randint(max(lo + 1, hi_min), hi_max)
        ammo_range[k] = (lo, hi)

    return target_counts, weapon_dist, ammo_range

# ── Geometry helpers (all distances in km) ────────────────────────────────────

def place_at_km(origin_x, origin_y, range_km, azimuth_deg):
    """Return (x, y) km offset from origin at given range and azimuth."""
    az = math.radians(azimuth_deg)
    return (
        round(origin_x + range_km * math.sin(az), 4),
        round(origin_y + range_km * math.cos(az), 4),
    )

def bearing_to_km(fx, fy, tx, ty):
    """Azimuth (deg, 0=North CW) from point f to point t."""
    return math.degrees(math.atan2(tx - fx, ty - fy)) % 360

def random_az_in_sector(az_from, az_to):
    if az_from <= az_to:
        return random.uniform(az_from, az_to)
    span = (az_to + 360) - az_from
    return (az_from + random.uniform(0, span)) % 360

def alt_compatible(winfo, alt_km):
    alt_m = alt_km * KM2M
    return winfo["MinAltitude"] <= alt_m <= winfo["MaxAltitude"]

# ── Threat proximity filter ───────────────────────────────────────────────────
# A target is "proximity-threatening" to a vessel only if its predicted
# straight-line trajectory passes within this radius (km).
# ~3 km ≈ inner defensive bubble / CIWS effective range.
THREAT_RADIUS_KM = 3.0

def min_approach_km(target, vessel):
    """Minimum distance (km) between target's straight-line trajectory and vessel."""
    dx0 = target["X"] - vessel["X"]
    dy0 = target["Y"] - vessel["Y"]
    dz0 = target["Z"] - vessel["Z"]
    spd_tgt  = target["Speed"] * M2KM
    v_vessel = vessel["Speed"]  * M2KM
    vx = target["VX"] * spd_tgt - v_vessel * vessel["HeadingX"]
    vy = target["VY"] * spd_tgt - v_vessel * vessel["HeadingY"]
    vz = target["VZ"] * spd_tgt  # HeadingZ = 0 for surface ships
    v2 = vx**2 + vy**2 + vz**2
    if v2 < 1e-12:
        return math.sqrt(dx0**2 + dy0**2 + dz0**2)
    t_star = -(dx0*vx + dy0*vy + dz0*vz) / v2
    if t_star < 0:
        return math.sqrt(dx0**2 + dy0**2 + dz0**2)
    dx = dx0 + vx * t_star
    dy = dy0 + vy * t_star
    dz = dz0 + vz * t_star
    return math.sqrt(dx**2 + dy**2 + dz**2)

# ── Engagement window (computed from trajectory) ───────────────────────────────

def compute_engagement_window(vessel, winfo, target, t_max=60.0):
    """
    Analytically find [a_ij, b_ij] in seconds.

    The target trajectory is a parametric line in the vessel's reference frame:
        P(t) = P0 + t * V_rel,   where V_rel = V_target - V_vessel

    We find every t where P(t) crosses a boundary surface:
      1. Two spheres  |P|² = R²          — slant range (MinRange, MaxRange)
      2. Two half-planes  dx·cosθ = dy·sinθ  — azimuth sector boundaries
      3. Two elevation cones  dz² = tan²(el)·(dx²+dy²)  — elevation limits
      4. Two horizontal planes  dz = alt_bound  — altitude limits

    Boundary crossings partition [0, t_max] into sub-intervals in which every
    constraint is monotone, so checking one midpoint per interval is exact.
    """
    eps = 1e-9

    # ── Relative initial position (km) ────────────────────────────────────────
    dx0 = target["X"] - vessel["X"]
    dy0 = target["Y"] - vessel["Y"]
    dz0 = target["Z"] - vessel["Z"]

    # ── Relative velocity: V_target − V_vessel (km/s) ─────────────────────────
    spd_tgt  = target["Speed"] * M2KM          # target speed in km/s
    v_vessel = vessel["Speed"] * M2KM          # vessel speed in km/s
    vx = target["VX"] * spd_tgt - v_vessel * vessel["HeadingX"]
    vy = target["VY"] * spd_tgt - v_vessel * vessel["HeadingY"]
    vz = target["VZ"] * spd_tgt - v_vessel * vessel["HeadingZ"]   # HeadingZ=0 for ships

    # ── Weapon envelope in km ─────────────────────────────────────────────────
    r_min = winfo["MinRange"]    * M2KM
    r_max = winfo["MaxRange"]    * M2KM
    a_min = winfo["MinAltitude"] * M2KM
    a_max = winfo["MaxAltitude"] * M2KM

    def quadratic_roots(A, B, C):
        """Real roots of A·t²+B·t+C=0 inside [0, t_max]."""
        roots = []
        if abs(A) > eps:
            disc = B**2 - 4*A*C
            if disc >= 0:
                sq = math.sqrt(disc)
                for t in ((-B - sq) / (2*A), (-B + sq) / (2*A)):
                    if 0.0 <= t <= t_max:
                        roots.append(t)
        elif abs(B) > eps:              # degenerate linear case
            t = -C / B
            if 0.0 <= t <= t_max:
                roots.append(t)
        return roots

    candidates = [0.0, t_max]

    # 1. Sphere intersections: |P(t)|² = R²  (3-D slant range)
    #    |V|²·t² + 2(P0·V)·t + (|P0|²−R²) = 0
    A3 = vx**2 + vy**2 + vz**2
    B3 = 2.0 * (dx0*vx + dy0*vy + dz0*vz)
    C3 = dx0**2 + dy0**2 + dz0**2
    for R in (r_min, r_max):
        candidates += quadratic_roots(A3, B3, C3 - R**2)

    # 2. Azimuth half-planes through Z-axis: dx·cos(θ) − dy·sin(θ) = 0  →  linear
    #    t = −(dx0·cosθ − dy0·sinθ) / (vx·cosθ − vy·sinθ)
    for az_deg in (winfo["AzimuthFromDeg"], winfo["AzimuthToDeg"]):
        th    = math.radians(az_deg)
        denom = vx*math.cos(th) - vy*math.sin(th)
        if abs(denom) > eps:
            t = -(dx0*math.cos(th) - dy0*math.sin(th)) / denom
            if 0.0 <= t <= t_max:
                candidates.append(t)

    # 3. Elevation cones: dz² = tan²(el)·(dx²+dy²)  →  quadratic
    #    [vz²−tan²·(vx²+vy²)]·t² + 2[dz0·vz−tan²·(dx0·vx+dy0·vy)]·t
    #     + [dz0²−tan²·(dx0²+dy0²)] = 0
    for el_deg in (winfo["ElevationMinDeg"], winfo["ElevationMaxDeg"]):
        tan_el = math.tan(math.radians(el_deg))
        t2     = tan_el**2
        Ah = vx**2 + vy**2
        Bh = dx0*vx + dy0*vy
        Ch = dx0**2 + dy0**2
        candidates += quadratic_roots(vz**2 - t2*Ah,
                                      2*(dz0*vz - t2*Bh),
                                      dz0**2   - t2*Ch)

    # 4. Altitude planes: dz(t) = alt_bound  →  linear
    if abs(vz) > eps:
        for ab in (a_min, a_max):
            t = (ab - dz0) / vz
            if 0.0 <= t <= t_max:
                candidates.append(t)

    candidates = sorted(set(candidates))

    def in_envelope(t):
        dx   = dx0 + vx * t
        dy   = dy0 + vy * t
        dz   = dz0 + vz * t
        rng  = math.sqrt(dx**2 + dy**2 + dz**2)   # 3-D slant range
        rng_h = math.hypot(dx, dy)                  # horizontal range for az/el
        if rng_h < eps:
            return False
        az   = math.degrees(math.atan2(dx, dy)) % 360
        el   = math.degrees(math.atan2(dz, rng_h))
        af, at_ = winfo["AzimuthFromDeg"], winfo["AzimuthToDeg"]
        az_ok = (af <= az <= at_) if af <= at_ else (az >= af or az <= at_)
        return (
            r_min <= rng  <= r_max  and
            a_min <= dz   <= a_max  and
            az_ok                   and
            winfo["ElevationMinDeg"] <= el <= winfo["ElevationMaxDeg"]
        )

    a = None
    for i in range(len(candidates) - 1):
        mid = (candidates[i] + candidates[i + 1]) / 2.0
        if in_envelope(mid):
            if a is None:
                a = candidates[i]
            b = candidates[i + 1]
        else:
            if a is not None:
                if b - a >= 1.0:
                    return (round(a, 2), round(b, 2))  # first valid window
                a = None  # too short, keep looking
    if a is not None and b - a >= 1.0:
        return (round(a, 2), round(b, 2))

    return None

# ── Generators ─────────────────────────────────────────────────────────────────

def _heading_vec(azimuth_deg: float, pitch_deg: float = 0.0):
    """Unit vector (X=East, Y=North, Z=Up) from azimuth + pitch."""
    az  = math.radians(azimuth_deg)
    pit = math.radians(pitch_deg)
    cos_p = math.cos(pit)
    return (
        round(math.sin(az) * cos_p, 6),
        round(math.cos(az) * cos_p, 6),
        round(math.sin(pit),        6),
    )

def generate_vessels():
    # Vessel 1 is the origin (0, 0, 0).
    # Vessel 2 is ~0.55 km NNE of vessel 1.
    vessels = []
    for vid, x, y in [(1, 0.0, 0.0), (2, 0.3, 0.45)]:
        spd_ms = random.uniform(3.0, 8.0)          # m/s
        spd_km = round(spd_ms * M2KM, 5)           # km/s
        az     = random.uniform(0, 360)
        hx, hy, _ = _heading_vec(az)               # ships stay on surface: hz=0
        vessels.append(dict(
            ID=vid, X=x, Y=y, Z=0.0,
            Speed=round(spd_ms, 2),                # speed in m/s kept for reference
            HeadingX=hx, HeadingY=hy, HeadingZ=0.0,
            DefenseRadius=25.0,                    # km
        ))
    return vessels

def generate_weapons(vessels, weapon_dist, ammo_range):
    weapons = []
    wpn_id = 1
    for vessel in vessels:
        for wcode, count in weapon_dist.items():
            ammo_lo, ammo_hi = ammo_range[wcode]
            for _ in range(count):
                weapons.append(dict(
                    ID=wpn_id,
                    WTAVesselID=vessel["ID"],
                    Ammo=random.randint(ammo_lo, ammo_hi),
                    WTAWeaponInfoCode=wcode,
                    Status=1,  # 1 = ready
                ))
                wpn_id += 1
    return weapons

def generate_targets(vessels, target_counts):
    targets = []
    tgt_id = 1

    for tcode, count in target_counts.items():
        props = TARGET_PROPS[tcode]
        for _ in range(count):
            spd_ms  = random.uniform(*props["speed"])        # m/s
            spd_km  = spd_ms * M2KM                          # km/s
            alt_m   = random.uniform(*props["alt"])          # metres
            alt_km  = alt_m * M2KM                           # km
            threat  = round(random.uniform(*props["threat"]), 3)

            # pick vessel and primary weapon with compatible altitude
            vessel  = random.choice(vessels)
            primary = [c for c in props["primary"] if alt_compatible(WI[c], alt_km)]
            if not primary:
                primary = props["primary"]
            wcode   = random.choice(primary)
            winfo   = WI[wcode]

            # place target outside max range so it enters within t_entry seconds
            t_entry      = random.uniform(0.5, 7.0)
            initial_range_km = winfo["MaxRange"] * M2KM + spd_km * t_entry

            # random azimuth within weapon sector
            az = random_az_in_sector(winfo["AzimuthFromDeg"], winfo["AzimuthToDeg"])
            x0, y0 = place_at_km(vessel["X"], vessel["Y"], initial_range_km, az)

            # heading toward vessel with miss distance encoded at generation time:
            #   80 % near-pass: miss < 3 km  →  only compute windows for these
            #   20 % far-pass:  miss > 3 km  →  proximity filter will drop them
            hdg_to_vessel = bearing_to_km(x0, y0, vessel["X"], vessel["Y"])
            near_pass = random.random() < 0.80
            if near_pass:
                miss_km = random.uniform(0.1, 2.5)   # guaranteed < THREAT_RADIUS_KM
            else:
                miss_km = random.uniform(3.5, 8.0)   # guaranteed > THREAT_RADIUS_KM
            deflect = math.degrees(math.asin(min(miss_km / initial_range_km, 1.0)))
            hdg = (hdg_to_vessel + random.choice([-1, 1]) * deflect) % 360
            # targets fly level (vz=0) except give tiny random pitch for realism
            pitch = random.uniform(*props["pitch"])  # type-specific pitch range
            pitch = 0.0 if alt_km <= 0.01 else pitch
            hx, hy, hz = _heading_vec(hdg, pitch)

            targets.append(dict(
                ID=tgt_id,
                WTATargetInfoCode=tcode,
                X=x0, Y=y0, Z=round(alt_km, 4),
                VX=hx,                       # unit heading vector
                VY=hy,                       # |VX,VY,VZ| = 1
                VZ=hz,
                Speed=round(spd_ms, 2),      # m/s; actual velocity = Speed * M2KM * (VX,VY,VZ)
                ThreatScore=threat,
            ))
            tgt_id += 1

    random.shuffle(targets)
    for i, t in enumerate(targets):
        t["ID"] = i + 1
    return targets

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default=None,   help="Single output file (overrides --outdir/--count)")
    parser.add_argument("--outdir", default="data", help="Output directory for batch generation")
    parser.add_argument("--count",  type=int, default=1, help="Number of scenarios to generate")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    # resolve output paths
    if args.out:
        out_paths = [args.out]
        seeds     = [args.seed]
    else:
        os.makedirs(args.outdir, exist_ok=True)
        out_paths = [os.path.join(args.outdir, f"scenario_{i+1:03d}.json") for i in range(args.count)]
        seeds     = [args.seed + i for i in range(args.count)]

    for out_path, seed in zip(out_paths, seeds):
        random.seed(seed)

        target_counts, weapon_dist, ammo_range = randomize_scenario_params()

        vessels  = generate_vessels()
        weapons  = generate_weapons(vessels, weapon_dist, ammo_range)
        targets  = generate_targets(vessels, target_counts)

        # build weapon lookup for window computation
        wpn_info_by_id = {w["ID"]: WI[w["WTAWeaponInfoCode"]] for w in weapons}
        vessel_by_id   = {v["ID"]: v for v in vessels}

        # compute engagement windows for every (weapon, target) pair
        # targets with a far-pass trajectory (min approach > THREAT_RADIUS_KM) are skipped
        windows = {}
        covered = set()
        for wpn in weapons:
            winfo  = wpn_info_by_id[wpn["ID"]]
            vessel = vessel_by_id[wpn["WTAVesselID"]]
            for tgt in targets:
                if min_approach_km(tgt, vessel) > THREAT_RADIUS_KM:
                    continue
                win = compute_engagement_window(vessel, winfo, tgt)
                if win is not None:
                    windows[f"{wpn['ID']}_{tgt['ID']}"] = win
                    covered.add(tgt["ID"])

        uncovered = [t["ID"] for t in targets if t["ID"] not in covered]
        n_covered = len(covered)
        warn = f"  WARNING: no window for targets {uncovered}" if uncovered else ""

        # strip internal-only fields before output
        for t in targets:
            t.pop("VerticalSpeed", None)

        output = dict(
            weapon_infos=WEAPON_INFOS,
            target_infos=TARGET_INFOS,
            probability_table=PROB_TABLE,
            assignment_request=dict(
                vessels=vessels,
                weapons=weapons,
                targets=targets,
            ),
            engagement_windows=windows,
        )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[seed={seed}] {out_path}  ({n_covered}/100 targets covered){warn}")

if __name__ == "__main__":
    main()