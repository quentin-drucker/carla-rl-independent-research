"""Pure tracking for commanded versus physically completed route recovery."""

from dataclasses import dataclass


@dataclass
class PhysicalRouteReturnTracker:
    """Confirm that the ego itself returned near route center after departing.

    A lateral-offset controller reaching the end of its return schedule only
    proves that its *request* reached zero. Physical recovery additionally
    requires the measured route-relative ego offset to remain near zero for a
    stable number of ticks after the ego previously left that region.
    """

    return_tolerance_m: float = 0.25
    required_return_ticks: int = 10
    departed_route: bool = False
    consecutive_return_ticks: int = 0
    physically_returned: bool = False
    final_actual_offset_m: float | None = None

    def __post_init__(self):
        if self.return_tolerance_m <= 0.0 or self.required_return_ticks <= 0:
            raise ValueError("return tolerance/ticks must be positive")

    def update(self, *, requested_offset_m, actual_offset_m) -> bool:
        if requested_offset_m is None or actual_offset_m is None:
            return self.physically_returned

        requested_offset_m = float(requested_offset_m)
        actual_offset_m = float(actual_offset_m)
        self.final_actual_offset_m = actual_offset_m

        if abs(actual_offset_m) > self.return_tolerance_m:
            self.departed_route = True

        physically_centered = (
            self.departed_route
            and abs(requested_offset_m) <= self.return_tolerance_m
            and abs(actual_offset_m) <= self.return_tolerance_m
        )
        if physically_centered:
            self.consecutive_return_ticks += 1
            if self.consecutive_return_ticks >= self.required_return_ticks:
                self.physically_returned = True
        else:
            self.consecutive_return_ticks = 0
            # This is a final-state metric, not an "ever crossed center"
            # event. A later departure invalidates an earlier stable return.
            self.physically_returned = False

        return self.physically_returned
