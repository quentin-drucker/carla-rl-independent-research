"""
avoidability.py
Quentin | Junior Spring Research, 2026.

Computes whether a collision is physically avoidable given a ScenarioConfig.

Used to distinguish between:
  - Agent failure          (collision was avoidable but agent didn't brake correctly)
  - Borderline             (within BORDERLINE_MARGIN_M of the physics limit — ambiguous)
  - Physically impossible  (no braking strategy could have avoided collision)

PHYSICS MODEL
-------------
When the trigger fires, the ego is traveling at v0 m/s and has trigger_ttc_s seconds
before reaching the crossing point:

    d_available = v0 * trigger_ttc_s
    d_stop      = v0² / (2 * A_MAX_DEFAULT)
    margin_m    = d_available - d_stop   (positive = room to spare, negative = overrun)

Applies to all scenarios regardless of pedestrian crossing mode. The ramp limiter
overhead is already embedded in A_MAX_DEFAULT (3.5 m/s²), back-calculated from
actual sweep stop outcomes through the full controller pipeline.

BORDERLINE ZONE
---------------
Scenarios within BORDERLINE_MARGIN_M = 8.0m of the threshold are flagged as
borderline. Small calibration error in A_MAX_DEFAULT, ped timing variance, or
encounter distance jitter can flip the label either way in this zone.

CALIBRATION NOTE
----------------
A_MAX_DEFAULT empirically derived via two-step process:
  1. brake_calibration.py (direct apply_control):   a_max_vehicle  = 6.96-7.56 m/s²
  2. validate_avoidability.py (through controller): a_max_effective = 3.04-4.02 m/s²
The large gap is the ramp limiter. Using 3.5 m/s² (mean of implied values, 20/24 correct).
"""

from scenario_config import ScenarioConfig

A_MAX_DEFAULT       = 3.5  # m/s²  effective system a_max, validated vs sweep data
BORDERLINE_MARGIN_M = 8.0  # m     margin band where label is uncertain
FAR_CROSS_LANE_M    = 3.3  # m     single lane width (Town04, user-measured)
                            #       far-cross ped must walk this far to clear ego's lane


def compute_avoidability(
    cfg: ScenarioConfig,
    a_max_mps2: float = A_MAX_DEFAULT,
) -> dict:
    """
    Determine whether a collision is physically avoidable for a given ScenarioConfig.

    Returns dict with keys:
        avoidable   (bool)  — True if physics allow stopping before crossing
        borderline  (bool)  — True if |margin| < BORDERLINE_MARGIN_M
        label       (str)   — one of: "avoidable", "borderline-avoidable",
                              "borderline-impossible", "impossible"
        margin_m    (float) — d_available - d_stop (m). Positive = room to spare,
                              negative = overrun. inf if far-cross ped cleared.
        d_available_m (float) — distance to crossing when trigger fires (m)
        d_stop_m    (float) — minimum stopping distance (m)
        v0_mps      (float) — ego speed at trigger (m/s)
        note        (str)   — human-readable explanation
    """
    v0  = cfg.target_mph * 0.44704  # mph -> m/s
    ttc = cfg.trigger_ttc_s if cfg.trigger_ttc_s is not None else (cfg.walker_startup_s + 1.5)

    # Far-cross clearing check: if the ped finishes crossing before ego arrives,
    # there is no collision threat regardless of speed — always avoidable.
    # Uses "avoidable" label (not a separate category) to keep 4-label system.
    if cfg.walker_cross.lower() == "far":
        time_to_clear = FAR_CROSS_LANE_M / cfg.walker_speed_mps
        if ttc >= time_to_clear:
            return {
                "avoidable":     True,
                "borderline":    False,
                "label":         "avoidable",
                "margin_m":      float("inf"),
                "d_available_m": float("inf"),
                "d_stop_m":      0.0,
                "v0_mps":        v0,
                "note":          (f"far-cross: ped clears in {time_to_clear:.1f}s, "
                                  f"ego arrives in {ttc:.1f}s — no threat"),
            }

    d_available = v0 * ttc
    d_stop      = (v0 ** 2) / (2.0 * a_max_mps2)
    margin      = d_available - d_stop
    avoidable   = margin >= 0.0
    borderline  = abs(margin) < BORDERLINE_MARGIN_M

    if avoidable and borderline:
        label = "borderline-avoidable"
        note  = (f"borderline-avoidable: {margin:.1f}m margin "
                 f"(within {BORDERLINE_MARGIN_M:.0f}m uncertainty band)")
    elif avoidable:
        label = "avoidable"
        note  = f"avoidable: {margin:.1f}m to spare"
    elif borderline:
        label = "borderline-impossible"
        note  = (f"borderline-impossible: overruns by {abs(margin):.1f}m "
                 f"(within {BORDERLINE_MARGIN_M:.0f}m uncertainty band)")
    else:
        label = "impossible"
        note  = f"IMPOSSIBLE: overruns crossing by {abs(margin):.1f}m at full brake"

    return {
        "avoidable":     avoidable,
        "borderline":    borderline,
        "label":         label,
        "margin_m":      margin,
        "d_available_m": d_available,
        "d_stop_m":      d_stop,
        "v0_mps":        v0,
        "note":          note,
    }


def avoidability_label(cfg: ScenarioConfig, a_max_mps2: float = A_MAX_DEFAULT) -> str:
    """
    Short label for print output and plot annotations.
    Returns one of: "avoidable", "borderline-avoidable",
    "borderline-impossible", "impossible".
    """
    return compute_avoidability(cfg, a_max_mps2)["label"]
