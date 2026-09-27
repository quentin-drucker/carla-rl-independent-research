"""reactive_avoidance.py

Computes the swerve offset actually needed to clear a pedestrian at a
known lateral position, instead of guessing a fixed constant (the root
cause of test16's undetected collision, 2026-09-27 -- a peak_offset_m of
1.5m was picked without reference to where the pedestrian actually ends
up, and happened to land almost exactly on the hazard corridor's own
blind-spot boundary).

Pure geometry, no CARLA imports -- offline-testable before any live use,
same discipline as vehicle_occupancy.py / pedestrian_contact.py.

Scope (2026-09-27, first pass): the STATIONARY-pedestrian case, where the
pedestrian's lateral position is fixed and known in advance, so a single
correctly-computed offset (fed into the existing, unmodified
build_evasive_offset_fn) is sufficient -- no new stateful controller
needed. A pedestrian who is still MOVING when the offset is computed
(e.g. test16's far-crossing case) is a harder, explicitly NOT-yet-solved
follow-up: this module's function only reflects the pedestrian's position
at the moment it's called, and does not itself re-check or widen the
offset later if the pedestrian keeps moving after that.
"""

from typing import Literal

Side = Literal[-1, 1]


def compute_required_clearance_offset_m(
    *,
    pedestrian_lateral_m: float,
    side_sign: float,
    ego_half_width_m: float = 1.082,
    pedestrian_radius_m: float = 0.3,
    safety_margin_m: float = 0.3,
) -> float:
    """The signed route-relative lateral offset the ego must reach, passing
    the pedestrian on the given side, to clear them with margin.

    side_sign: +1 to pass on route-right of the pedestrian, -1 to pass on
    route-left. (Sign, not magnitude -- only >=0 vs <0 is checked.)

    clearance = ego_half_width_m + pedestrian_radius_m + safety_margin_m is
    the minimum center-to-center lateral separation once the ego has
    reached the returned offset. This mirrors pedestrian_contact.py's
    conservative-radius reasoning but as a MINIMUM separation to aim for,
    not a contact threshold to detect after the fact.

    Only correct for a pedestrian whose lateral position is not expected to
    change further before the ego reaches them -- see module docstring.
    """
    clearance_m = ego_half_width_m + pedestrian_radius_m + safety_margin_m
    if side_sign >= 0:
        return pedestrian_lateral_m + clearance_m
    return pedestrian_lateral_m - clearance_m
