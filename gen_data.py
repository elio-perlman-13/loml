#!/usr/bin/env python3
"""
WTA Scenario Data Generator
Outputs JSON matching CMS_WTA.proto schema (WTAAssignmentRequest + catalogs).
"""

import json
import math
import random
import argparse

random.seed(42)

# ── Geo constants (at ~10°N) ───────────────────────────────────────────────────
LAT_M = 111_000.0
LON_M = 111_000.0 * math.cos(math.radians(10.5))

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
    "TGT_ASM":        dict(speed=(270, 310), alt=(3,   15),   threat=(0.85, 1.00),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD", "SAM_PK"]),
    # Fighter: high-alt, fast
    "TGT_FIGHTER":    dict(speed=(200, 320), alt=(500, 6000), threat=(0.65, 0.90),
                           primary=["SAM_PK"]),
    # Helicopter: slow, low
    "TGT_HELICOPTER": dict(speed=(25,  70),  alt=(20,  400),  threat=(0.35, 0.65),
                           primary=["SAM_PK", "AK630_PORT", "AK630_STBD"]),
    # Fixed-wing UAV
    "TGT_UAV_FIXED":  dict(speed=(30,  80),  alt=(100, 800),  threat=(0.45, 0.75),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Rotary UAV: very slow, low
    "TGT_UAV_ROTOR":  dict(speed=(5,   30),  alt=(10,  200),  threat=(0.40, 0.70),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Kamikaze drone: medium speed, low-mid alt
    "TGT_KAMIKAZE":   dict(speed=(50, 130),  alt=(20,  500),  threat=(0.55, 0.85),
                           primary=["TCDT_BV15", "AK630_PORT", "AK630_STBD"]),
    # Surface warship: slow, at sea level
    "TGT_SURFACE":    dict(speed=(8,   20),  alt=(0,   5),    threat=(0.60, 0.95),
                           primary=["VCM", "AK176"]),
    # USV: small, fast unmanned surface vessel
    "TGT_USV":        dict(speed=(15,  35),  alt=(0,   3),    threat=(0.40, 0.70),
                           primary=["AK176", "AK630_PORT", "AK630_STBD"]),
    # Torpedo: very fast underwater – treat altitude as slightly negative
    "TGT_TORPEDO":    dict(speed=(20,  40),  alt=(-3,  0),    threat=(0.50, 0.80),
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

# ── Geometry helpers ───────────────────────────────────────────────────────────

def place_at(vessel_lat, vessel_lon, range_m, azimuth_deg):
    """Return (lat, lon) at given range and azimuth from vessel."""
    az = math.radians(azimuth_deg)
    lat = vessel_lat + (range_m * math.cos(az)) / LAT_M
    lon = vessel_lon + (range_m * math.sin(az)) / LON_M
    return lat, lon

def bearing_to(from_lat, from_lon, to_lat, to_lon):
    dy = (to_lat - from_lat) * LAT_M
    dx = (to_lon - from_lon) * LON_M
    return math.degrees(math.atan2(dx, dy)) % 360

def random_az_in_sector(az_from, az_to):
    if az_from <= az_to:
        return random.uniform(az_from, az_to)
    span = (az_to + 360) - az_from
    return (az_from + random.uniform(0, span)) % 360

def alt_compatible(winfo, alt):
    return winfo["MinAltitude"] <= alt <= winfo["MaxAltitude"]

# ── Engagement window (computed from trajectory) ───────────────────────────────

def compute_engagement_window(vessel, winfo, target, dt=0.1, t_max=120.0):
    """
    Numerically sweep target trajectory to find [a_ij, b_ij].
    Returns (a, b) or None if target never enters envelope.
    """
    vx = vessel["Longitude"];  vy = vessel["Latitude"]
    tx0 = target["Longitude"]; ty0 = target["Latitude"]; alt0 = target["Altitude"]
    spd = target["Speed"];     hdg = math.radians(target["Heading"])

    v_spd = target.get("VerticalSpeed", 0.0)

    a, b = None, None
    t = 0.0
    while t <= t_max:
        # target position at time t
        tx = tx0 + (spd * math.sin(hdg) * t) / LON_M
        ty = ty0 + (spd * math.cos(hdg) * t) / LAT_M
        alt = alt0 + v_spd * t

        dx = (tx - vx) * LON_M
        dy = (ty - vy) * LAT_M
        rng = math.hypot(dx, dy)
        az  = math.degrees(math.atan2(dx, dy)) % 360
        el  = math.degrees(math.atan2(alt, rng)) if rng > 0 else 90.0

        # azimuth check (handles wrap-around)
        af, at_ = winfo["AzimuthFromDeg"], winfo["AzimuthToDeg"]
        if af <= at_:
            az_ok = af <= az <= at_
        else:
            az_ok = az >= af or az <= at_

        in_envelope = (
            winfo["MinRange"]    <= rng <= winfo["MaxRange"]    and
            winfo["MinAltitude"] <= alt <= winfo["MaxAltitude"] and
            az_ok and
            winfo["ElevationMinDeg"] <= el <= winfo["ElevationMaxDeg"]
        )

        if in_envelope and a is None:
            a = t
        elif not in_envelope and a is not None:
            b = t - dt
            break

        t += dt

    if a is not None and b is None:
        b = t_max  # still in envelope at end of horizon
    return (round(a, 2), round(b, 2)) if a is not None else None

# ── Generators ─────────────────────────────────────────────────────────────────

def _heading_vec(azimuth_deg: float, pitch_deg: float = 0.0):
    """Convert azimuth (0=North, CW) + pitch (deg, positive=up) to a
    normalised 3-D unit vector in (East, North, Up) frame."""
    az  = math.radians(azimuth_deg)
    pit = math.radians(pitch_deg)
    cos_p = math.cos(pit)
    return {
        "x": round(math.sin(az) * cos_p, 6),   # East
        "y": round(math.cos(az) * cos_p, 6),   # North
        "z": round(math.sin(pit),        6),   # Up
    }

def generate_vessels():
    vessels = []
    configs = [
        dict(ID=1, Latitude=10.5000, Longitude=107.5000, Altitude=0.0,
             Speed=round(random.uniform(3.0, 8.0), 2),
             HeadingDeg=round(random.uniform(0, 360), 1),
             DefenseRadius=25_000.0),
        dict(ID=2, Latitude=10.5045, Longitude=107.5030, Altitude=0.0,
             Speed=round(random.uniform(3.0, 8.0), 2),
             HeadingDeg=round(random.uniform(0, 360), 1),
             DefenseRadius=25_000.0),
    ]
    for cfg in configs:
        az  = cfg.pop("HeadingDeg")
        hvec = _heading_vec(az)   # surface ship: pitch = 0
        vessels.append({**cfg, "HeadingX": hvec["x"], "HeadingY": hvec["y"], "HeadingZ": hvec["z"]})
    return vessels  # ~550 m apart

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
            speed   = random.uniform(*props["speed"])
            alt     = random.uniform(*props["alt"])
            threat  = round(random.uniform(*props["threat"]), 3)

            # pick vessel and primary weapon ensuring altitude compatibility
            vessel  = random.choice(vessels)
            primary = [c for c in props["primary"] if alt_compatible(WI[c], alt)]
            if not primary:
                primary = props["primary"]  # fallback (may be edge case)
            wcode   = random.choice(primary)
            winfo   = WI[wcode]

            # place target: slightly outside max range so it enters within t_entry seconds
            t_entry = random.uniform(0.5, 7.0)
            initial_range = winfo["MaxRange"] + speed * t_entry

            # random azimuth within weapon sector
            az = random_az_in_sector(winfo["AzimuthFromDeg"], winfo["AzimuthToDeg"])
            lat, lon = place_at(vessel["Latitude"], vessel["Longitude"], initial_range, az)

            # heading toward vessel ± 10° jitter
            hdg_to_vessel = bearing_to(lat, lon, vessel["Latitude"], vessel["Longitude"])
            heading = (hdg_to_vessel + random.uniform(-10, 10)) % 360

            targets.append(dict(
                ID=tgt_id,
                WTATargetInfoCode=tcode,
                Latitude=round(lat, 7),
                Longitude=round(lon, 7),
                Altitude=round(alt, 1),
                Speed=round(speed, 2),
                Heading=round(heading, 2),
                ThreatScore=threat,
                VerticalSpeed=0.0,  # extra field used by window computation; strip before sending
            ))
            tgt_id += 1

    random.shuffle(targets)
    # re-index after shuffle
    for i, t in enumerate(targets):
        t["ID"] = i + 1
    return targets

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/scenario_001.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    target_counts, weapon_dist, ammo_range = randomize_scenario_params()

    vessels  = generate_vessels()
    weapons  = generate_weapons(vessels, weapon_dist, ammo_range)
    targets  = generate_targets(vessels, target_counts)

    # build weapon lookup for window computation
    wpn_info_by_id = {w["ID"]: WI[w["WTAWeaponInfoCode"]] for w in weapons}
    vessel_by_id   = {v["ID"]: v for v in vessels}

    # compute engagement windows for every (weapon, target) pair
    print("Computing engagement windows...")
    windows = {}
    covered = set()
    for wpn in weapons:
        winfo  = wpn_info_by_id[wpn["ID"]]
        vessel = vessel_by_id[wpn["WTAVesselID"]]
        for tgt in targets:
            win = compute_engagement_window(vessel, winfo, tgt)
            if win is not None:
                windows[f"{wpn['ID']}_{tgt['ID']}"] = win
                covered.add(tgt["ID"])

    uncovered = [t["ID"] for t in targets if t["ID"] not in covered]
    print(f"  {len(covered)}/100 targets have at least 1 engagement window")
    if uncovered:
        print(f"  WARNING: targets with no window: {uncovered}")

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
        engagement_windows=windows,  # precomputed; solver can recompute if needed
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Written to {args.out}")

if __name__ == "__main__":
    main()