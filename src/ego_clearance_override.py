"""ego_clearance_override.py

A geometric, ego-position-based override for original-lane braking authority.

Background: test17 (2026-09-27) found that original-lane braking never looks
at where the ego ACTUALLY is -- only whether the original lane was ever
occupied -- so a correctly-swerved-around pedestrian still forces a full
stop. Quentin asked to fix this by having "the corridor move with the ego,"
which is literally the twice-reverted, flagged Non-goal (letting steering
suppress original-lane braking via a corridor-reading swap). This module is
a DIFFERENT mechanism for the same category of change: instead of trusting a
second LiDAR corridor's reading (the thing that failed twice -- false
clearance from sparse returns, then governing-source chatter), it computes
the straight-line, ANY-direction distance between the ego's OWN current
position and the pedestrian's OWN current position, both ground truth, live
every tick. It is still a real reduction of original-lane braking authority
and carries the same risk category -- built only after explicit sign-off.

STATUS (2026-09-27): experimental, NOT validated -- do not treat any variant
below as a working fix. Three live-tested iterations so far, in order:

Attempt 1 used a LATERAL-ONLY projection (perpendicular distance from the
ego's heading axis) instead of full euclidean distance. Live-tested (test18)
and its recorded trace showed 21 ticks flagged as "contact" by
pedestrian_contact.py's ground-truth check, at a genuine lateral separation
of ~1.68m (matching the offset commanded by reactive_avoidance.py). NOT
YET DETERMINED whether this is a real body-to-body hit or a false alarm:
pedestrian_contact.py's contact radius uses max(ego_half_length_m,
ego_half_width_m) = 2.396m for ANY-direction contact (deliberately
conservative near the front/rear corners), which is much larger than
ego_half_width_m = 1.082m -- the number that actually matters for a
strictly-lateral, directly-beside pass. A 1.68m lateral gap may well be
genuinely safe (1.68 - 1.082 - 0.3(ped radius) = ~0.3m real clearance) and
still trip the conservative circular check. Distinguishing "real collision"
from "conservative-check false alarm" needs an oriented-rectangle-vs-circle
contact model (using the ego's actual heading), which pedestrian_contact.py
does not currently do -- NOT built this session.

Attempt 2 switched to full circular euclidean distance (matching
pedestrian_contact.py's own radius), reasoning that it could not have
attempt 1's directional blind spot. Live-tested and found a DIFFERENT bug:
distance alone cannot tell "35m away and still approaching" apart from
"already passed and clear" -- both read as a large distance. The override
falsely confirmed clearance from the very first tick after trigger and
disabled hazard braking for the entire approach in BOTH test18 cases,
including the undersized-offset case that should have braked.

Attempt 3 (current code) adds is_pedestrian_behind_ego() as a required
condition alongside the circular-distance check. Live-tested and found
SAFE (0 ground-truth contact ticks in both test18 cases) but FUNCTIONALLY
INERT for the scenario it was built for: requiring "pedestrian already
behind ego" means the override can only ever engage after the ego has
fully passed the pedestrian -- for a lane-center stationary pedestrian, the
UNTOUCHED original-lane corridor brakes the ego to a stop well before it
gets that far (its trigger distance, headway-seconds * speed, is much
larger than the pedestrian's actual proximity). Both test18 cases came back
full_stop again, identical to test17 -- no "swerve without unnecessary
braking" behavior has been demonstrated yet by any of the three attempts.

Not done this session: an oriented-rectangle contact/clearance model (using
ego heading, not just a circular radius or a fixed forward/behind cutoff)
that could resolve BOTH open problems at once -- confirming genuine lateral
clearance while still approaching (needed to avoid the full-stop-before-
passing problem) without attempt 1's apparent false-alarm risk. This is
flagged for explicit follow-up, not attempted here.

Pure geometry, no CARLA imports -- offline-testable before any live use,
same discipline as vehicle_occupancy.py / pedestrian_contact.py /
reactive_avoidance.py.
"""

import math

from pedestrian_contact import compute_ego_pedestrian_contact_radius_m

# Extra margin on top of the contact-detection radius itself: "confirmed
# clear" must mean the ego has cleared by more than the bare non-contact
# boundary, not just barely missed it. Same pattern as hazard_governance.py
# / lane_follow.py's HAZARD_CLEAR_MARGIN_M (require positive margin above a
# threshold, not equality with it).
DEFAULT_CLEAR_MARGIN_M = 1.0


def compute_ego_pedestrian_distance_m(
    *,
    ego_x_m: float,
    ego_y_m: float,
    pedestrian_x_m: float,
    pedestrian_y_m: float,
) -> float:
    """Straight-line distance between the ego's and pedestrian's current
    ground-truth positions, in any direction (not projected onto a heading
    axis) -- see module docstring for why direction-agnostic matters here.
    """
    return math.hypot(pedestrian_x_m - ego_x_m, pedestrian_y_m - ego_y_m)


def is_pedestrian_behind_ego(
    *,
    ego_x_m: float,
    ego_y_m: float,
    ego_forward_x: float,
    ego_forward_y: float,
    pedestrian_x_m: float,
    pedestrian_y_m: float,
) -> bool:
    """True if the pedestrian is behind the ego's current heading (negative
    dot product with the forward vector).

    Distance alone (compute_ego_pedestrian_distance_m) cannot tell "already
    passed and clear" apart from "still far away and approaching" -- both
    read as a large distance. Live-tested (test18, first fix attempt) and
    found to falsely confirm clearance from the very first tick after
    trigger, while the pedestrian was still 35m AHEAD, permanently
    suppressing hazard braking for the whole approach. Requiring the
    pedestrian to be behind the ego closes that gap.
    """
    dx = pedestrian_x_m - ego_x_m
    dy = pedestrian_y_m - ego_y_m
    return (dx * ego_forward_x + dy * ego_forward_y) < 0.0


def is_ego_geometrically_clear_of_pedestrian(
    *,
    distance_m: float,
    required_clearance_m: float = None,
) -> bool:
    """True only if the ego's current actual distance from the pedestrian
    already exceeds the minimum clearance to call it "clear".

    required_clearance_m: pass None to use
    compute_ego_pedestrian_contact_radius_m(safety_margin_m=DEFAULT_CLEAR_MARGIN_M)
    -- the same conservative circular footprint pedestrian_contact.py uses
    to detect real contact, plus a margin, so "clear" here means genuinely
    past the boundary that would otherwise be flagged as contact.
    """
    if required_clearance_m is None:
        required_clearance_m = compute_ego_pedestrian_contact_radius_m(
            safety_margin_m=DEFAULT_CLEAR_MARGIN_M
        )
    return distance_m > required_clearance_m
