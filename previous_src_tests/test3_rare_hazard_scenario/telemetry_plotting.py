"""
telemetry_plotting.py

Goal:
- Record telemetry each tick and plot after the simulation ends.
- Keep this simple + reliable (no real-time plot overhead by default).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import matplotlib.pyplot as plt


MPH_PER_MPS = 2.2369362920544  # 1 m/s = 2.2369 mph


def _safe_get(d: Optional[Dict[str, Any]], key: str, default=float("nan")):
    if d is None:
        return default
    v = d.get(key, default)
    if v is None:
        return default
    return v


@dataclass
class TelemetryBuffer:
    """
    Stores per-tick telemetry so we can plot it later.

    Notes:
    - If telemetry is None that tick, we store NaNs so plots show gaps.
    - We keep tick index and time (seconds) for x-axis.
    """
    t_ticks: List[int] = field(default_factory=list)
    t_sec: List[float] = field(default_factory=list)

    speed_mps: List[float] = field(default_factory=list)
    speed_error_mps: List[float] = field(default_factory=list)

    steer_cmd: List[float] = field(default_factory=list)
    throttle_cmd: List[float] = field(default_factory=list)
    brake_cmd: List[float] = field(default_factory=list)

    d_min_ahead_m: List[float] = field(default_factory=list)
    hazard_brake_cmd: List[float] = field(default_factory=list)
    brake_prev_state: List[float] = field(default_factory=list)

    # recently added (from lane_follow_step telemetry)
    trigger_distance_m: List[float] = field(default_factory=list)
    brake_target: List[float] = field(default_factory=list)

    cte_m: List[float] = field(default_factory=list)
    heading_error_rad: List[float] = field(default_factory=list)

    cone_yaw_deg: List[float] = field(default_factory=list)
    cone_half_angle_deg: List[float] = field(default_factory=list)

    jerk_mps3: List[float] = field(default_factory=list)
    ttc_s: List[float] = field(default_factory=list)

    def append(self, tick: int, fixed_dt: float, telemetry: Optional[Dict[str, Any]]):
        self.t_ticks.append(tick)
        self.t_sec.append(tick * fixed_dt)

        self.speed_mps.append(_safe_get(telemetry, "speed_mps"))
        self.speed_error_mps.append(_safe_get(telemetry, "speed_error_mps"))

        self.steer_cmd.append(_safe_get(telemetry, "steer_cmd"))
        self.throttle_cmd.append(_safe_get(telemetry, "throttle_cmd"))
        self.brake_cmd.append(_safe_get(telemetry, "brake_cmd"))

        self.d_min_ahead_m.append(_safe_get(telemetry, "d_min_ahead_m"))
        self.hazard_brake_cmd.append(_safe_get(telemetry, "hazard_brake_cmd"))
        self.brake_prev_state.append(_safe_get(telemetry, "brake_prev_state"))

        # New (if not present for older runs, this will be NaN)
        self.trigger_distance_m.append(_safe_get(telemetry, "trigger_distance_m"))
        self.brake_target.append(_safe_get(telemetry, "brake_target"))

        self.cte_m.append(_safe_get(telemetry, "cte_m"))
        self.heading_error_rad.append(_safe_get(telemetry, "heading_error_rad"))

        self.cone_yaw_deg.append(_safe_get(telemetry, "cone_yaw_deg"))
        self.cone_half_angle_deg.append(_safe_get(telemetry, "cone_half_angle_deg"))

        self.jerk_mps3.append(_safe_get(telemetry, "jerk_mps3"))
        self.ttc_s.append(_safe_get(telemetry, "ttc_s"))


def _mps_list_to_mph(vals_mps: List[float]) -> List[float]:
    out = []
    for v in vals_mps:
        out.append(v * MPH_PER_MPS if not math.isnan(v) else float("nan"))
    return out


def plot_telemetry(
    buffer: TelemetryBuffer,
    *,
    target_mph: float,
    # -----------------------------
    # Toggle plots (all are off by default)
    # -----------------------------
    show_speed: bool = False,           # basic speed tracking plot (speed vs target)
    show_braking_view: bool = False,    # LiDAR+braking tuning view with shaded hazard regions
    show_controls: bool = False,        # steer/throttle/brake commands overview
    show_lane_metrics: bool = False,    # lane following errors (cte and heading error)
    show_cone_params: bool = False,     # LiDAR hazard cone parameters (yaw and width)
    show_safety_metrics: bool = False,  # jerk (comfort) + TTC (risk) over time
    ):

    """
    Creates matplotlib windows. Close the windows to let the script exit.

    Behavior:
    - ONLY shows figures whose show_* flag is True.
    - Braking view uses hazard_active as a shaded region (not a 0/1 spike line).
    - Each figure includes a small blurb explaining what each line means.
    """
    t = buffer.t_sec
    if not t:
        print("plot_telemetry: buffer is empty (no telemetry recorded).")
        return

    # Slightly nicer defaults (no custom colors forced)
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    speed_mph = _mps_list_to_mph(buffer.speed_mps)

    # Decide trigger distance source:
    # Prefer telemetry trigger_distance_m (most faithful).
    # If it's all NaN, we simply won't plot trigger (unless you later add a fallback compute).
    trigger_dist = None
    have_trigger = any(not math.isnan(v) for v in buffer.trigger_distance_m)
    if have_trigger:
        trigger_dist = buffer.trigger_distance_m

    # Helper: add a text blurb in the figure (top-left margin area)
    def _add_blurb(fig, text: str):
        # NOTE: revised to place the blurb on a dedicated right-side panel (no overlap).
        # This function now expects the caller to pass a fig that already has a right-side text axis
        # OR to use _make_side_panel_figure() below.
        pass

    # Helper: create a figure with a right-side text panel (no overlap)
    def _make_side_panel_figure(title: str, nrows: int = 1, width_ratios=(4.5, 2.0), height_ratios=None):
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        fig.suptitle(title)

        if height_ratios is None:
            height_ratios = [1 for _ in range(nrows)]

        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=2,
            width_ratios=list(width_ratios),
            height_ratios=list(height_ratios)
        )

        axes_left = []
        for r in range(nrows):
            if r == 0:
                axes_left.append(fig.add_subplot(gs[r, 0]))
            else:
                axes_left.append(fig.add_subplot(gs[r, 0], sharex=axes_left[0]))

        ax_text = fig.add_subplot(gs[:, 1])  # right-side text panel spans all rows
        ax_text.axis("off")

        return fig, axes_left, ax_text

    # Helper: shade regions where hazard is active
    def _shade_hazard(ax, t_vals, hazard_vals):
        # hazard_vals expected to be 0/1 (or NaN). We'll shade wherever > 0.5.
        active = []
        for v in hazard_vals:
            if math.isnan(v):
                active.append(False)
            else:
                active.append(v > 0.5)

        # find contiguous active segments
        start = None
        for i in range(len(active)):
            if active[i] and start is None:
                start = i
            if (not active[i] or i == len(active) - 1) and start is not None:
                end = i if not active[i] else i  # include last point if active
                ax.axvspan(t_vals[start], t_vals[end], alpha=0.12, zorder=0)  # subtle shading BEHIND lines
                start = None

    # -------------------------
    # Figure: speed tracking
    # -------------------------
    if show_speed:
        fig1, axes_left, ax_text = _make_side_panel_figure("Speed Tracking", nrows=1)
        ax = axes_left[0]

        ax.plot(t, speed_mph, label="speed (mph)", linewidth=2.2, zorder=3)
        ax.plot(t, [target_mph for _ in t], label="target (mph)", linewidth=2.0, linestyle="--", zorder=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mph")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax_text.text(
            0.0, 1.0,
            "What this figure shows:\n\n"
            "• speed (mph): the vehicle's actual measured speed over time.\n"
            "• target (mph): the desired cruise speed set by your controller.\n\n"
            "Interpretation:\n"
            "This plot shows how well the longitudinal speed controller\n"
            "tracks the desired speed. Ideally, the solid speed line\n"
            "stays close to the dashed target line.\n\n"
            "If you see oscillations or overshoot, adjust KP/KI gains\n"
            "or reduce the target speed.",
            ha="left", va="top", fontsize=9
        )

    # -------------------------------------------------------
    # Figure: LiDAR + braking (tuning view)  <-- now opt-in
    # -------------------------------------------------------
    if show_braking_view:
        # --- 2-column layout: plots on the left, explanation panel on the right ---
        fig2, axes_left, ax_text = _make_side_panel_figure(
            "LiDAR + Braking (Tuning View)",
            nrows=3,
            height_ratios=[1.0, 1.0, 1.0]
        )

        ax1 = axes_left[0]  # Row 1: distance metrics
        ax2 = axes_left[1]  # Row 2: ramp internals
        ax3 = axes_left[2]  # Row 3: applied brake

        # --------------------------
        # Row 1: distance metrics
        # --------------------------
        ax1.plot(t, buffer.d_min_ahead_m, label="d_min_ahead (m)", linewidth=2.2, zorder=3)
        if trigger_dist is not None:
            ax1.plot(t, trigger_dist, label="trigger_distance (m)", linewidth=2.0, linestyle="--", zorder=2)

        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("meters")
        ax1.grid(True)
        ax1.legend(loc="upper right")
        _shade_hazard(ax1, t, buffer.hazard_brake_cmd)

        # --------------------------
        # Row 2: ramp internals (target vs ramp output)
        # --------------------------
        # Make overlapping lines easier to see:
        # - brake_prev_state: dashed, strong
        # - brake_target: dotted + markers, lighter
        ax2.plot(
            t, buffer.brake_prev_state,
            label="brake_prev_state (ramp output)",
            linewidth=2.4,
            linestyle="--",
            alpha=0.95,
            zorder=3
        )
        ax2.plot(
            t, buffer.brake_target,
            label="brake_target (ramp goal)",
            linewidth=1.8,
            linestyle=":",
            marker="o",
            markersize=2.5,
            alpha=0.75,
            zorder=2
        )
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("ramp (0..1)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True)
        ax2.legend(loc="upper right")
        _shade_hazard(ax2, t, buffer.hazard_brake_cmd)

        # --------------------------
        # Row 3: applied brake (what the vehicle actually received)
        # --------------------------
        ax3.plot(
            t, buffer.brake_cmd,
            label="brake_cmd (applied)",
            linewidth=2.6,
            linestyle="-",
            alpha=0.80,
            zorder=3
        )
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("brake (0..1)")
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True)
        ax3.legend(loc="upper right")
        _shade_hazard(ax3, t, buffer.hazard_brake_cmd)

        # --------------------------
        # Right-side explanation panel (NO OVERLAP)
        # --------------------------
        text = (
            "How to read this figure:\n\n"
            "Row 1 — Distance & Decision Boundary:\n"
            "• d_min_ahead (m): the closest LiDAR point detected\n"
            "  within the forward perception region.\n"
            "• trigger_distance (m): the braking activation distance\n"
            "  computed for this tick (distance where braking begins).\n"
            "• shaded regions: periods where LiDAR determined\n"
            "  the vehicle is too close to an obstacle.\n\n"
            "Row 2 — Braking Logic Internals:\n"
            "• brake_target: raw brake value computed directly\n"
            "  from obstacle distance.\n"
            "• brake_prev_state: smoothed brake value after\n"
            "  ramping (limits abrupt changes for comfort/stability).\n\n"
            "Row 3 — Applied Brake Command:\n"
            "• brake_cmd: final brake value sent to the vehicle.\n"
            "  This is what actually affects motion.\n\n"
            "How to use this for tuning:\n"
            "• If braking looks noisy → reduce ramp rate.\n"
            "• If braking is too late → increase headway/base distance.\n"
            "• If braking is too early → decrease headway/base distance."
        )
        ax_text.text(0.0, 1.0, text, ha="left", va="top", fontsize=9)

    # -----------------------------------------
    # Optional: steering + control commands
    # -----------------------------------------
    if show_controls:
        fig3, axes_left, ax_text = _make_side_panel_figure("Controls Overview", nrows=2, height_ratios=[1.0, 1.0])
        ax_top = axes_left[0]  # throttle + brake
        ax_bot = axes_left[1]  # steering

        # --------------------------
        # Top row: longitudinal control
        # --------------------------
        ax_top.plot(t, buffer.throttle_cmd, label="throttle_cmd", linewidth=2.2, zorder=3)
        ax_top.plot(t, buffer.brake_cmd, label="brake_cmd", linewidth=2.4, zorder=4, alpha=0.85)
        ax_top.set_xlabel("time (s)")
        ax_top.set_ylabel("throttle/brake (0..1)")
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.grid(True)
        ax_top.legend(loc="upper right")

        # --------------------------
        # Bottom row: steering
        # --------------------------
        ax_bot.plot(t, buffer.steer_cmd, label="steer_cmd", linewidth=2.2, zorder=3)
        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylabel("steer (-1..+1)")
        ax_bot.set_ylim(-1.05, 1.05)
        ax_bot.grid(True)
        ax_bot.legend(loc="upper right")

        ax_text.text(
            0.0, 1.0,
            "Controls Overview:\n\n"
            "Top Panel — Longitudinal Control:\n"
            "• throttle_cmd: throttle sent to vehicle (0 = none, 1 = full).\n"
            "• brake_cmd: brake sent to vehicle (0 = none, 1 = full).\n"
            "These should rarely be high at the same time.\n\n"
            "Bottom Panel — Steering:\n"
            "• steer_cmd: steering command\n"
            "  -1 = full LEFT, 0 = straight, +1 = full RIGHT.\n\n"
            "Use this plot to detect:\n"
            "• braking events during turns (possible false hazards)\n"
            "• throttle drops caused by braking overrides\n"
            "• command oscillations or instability",
            ha="left", va="top", fontsize=9
        )

    # -----------------------------------------
    # Optional: lane following errors
    # -----------------------------------------
    if show_lane_metrics:
        fig4, axes_left, ax_text = _make_side_panel_figure("Lane Metrics", nrows=1)
        ax = axes_left[0]

        ax.plot(t, buffer.cte_m, label="cte_m (m)", linewidth=2.2, zorder=3)
        ax.plot(t, buffer.heading_error_rad, label="heading_error (rad)", linewidth=2.0, zorder=2)
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax_text.text(
            0.0, 1.0,
            "Lane Following Errors:\n\n"
            "• cte_m: cross-track error (lateral distance\n"
            "  from the lane centerline).\n"
            "   + value = vehicle is right of centerline\n"
            "   - value = vehicle is left of centerline\n\n"
            "• heading_error_rad: angular difference between\n"
            "  vehicle heading and lane direction.\n"
            "   + value = vehicle pointing slightly right\n"
            "   - value = vehicle pointing slightly left\n\n"
            "Interpretation:\n"
            "Both values should stay near zero during stable\n"
            "lane following. Spikes typically occur in curves\n"
            "or when correcting from drift.",
            ha="left", va="top", fontsize=9
        )

    # -----------------------------------------
    # Optional: cone parameters
    # -----------------------------------------
    if show_cone_params:
        fig5, axes_left, ax_text = _make_side_panel_figure("Hazard Cone Params", nrows=1)
        ax = axes_left[0]

        ax.plot(t, buffer.cone_yaw_deg, label="cone_yaw_deg", linewidth=2.2, zorder=3)
        ax.plot(t, buffer.cone_half_angle_deg, label="cone_half_angle_deg", linewidth=2.0, zorder=2)
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax_text.text(
            0.0, 1.0,
            "LiDAR Hazard Perception Parameters:\n\n"
            "• cone_yaw_deg: direction the hazard region\n"
            "  is rotated relative to straight ahead.\n"
            "   + degrees = rotated toward the RIGHT\n"
            "   - degrees = rotated toward the LEFT\n"
            "   0° = straight forward.\n\n"
            "• cone_half_angle_deg: half-width of the\n"
            "  forward perception region.\n"
            "   Larger value = wider field of view.\n"
            "   Smaller value = narrower field of view.\n\n"
            "These parameters determine which LiDAR points\n"
            "are considered \"ahead\" of the vehicle and\n"
            "directly affect hazard detection during turns.",
            ha="left", va="top", fontsize=9
        )

    # -----------------------------------------
    # Optional: safety metrics (jerk + TTC)
    # -----------------------------------------
    if show_safety_metrics:
        fig6, axes_left, ax_text = _make_side_panel_figure(
            "Safety & Comfort Metrics", nrows=2, height_ratios=[1.0, 1.0]
        )
        ax_top = axes_left[0]  # jerk
        ax_bot = axes_left[1]  # TTC

        ax_top.plot(t, buffer.jerk_mps3, label="jerk (m/s³)", linewidth=1.8, zorder=3)
        ax_top.set_xlabel("time (s)")
        ax_top.set_ylabel("jerk (m/s³)")
        ax_top.grid(True)
        ax_top.legend(loc="upper right")
        _shade_hazard(ax_top, t, buffer.hazard_brake_cmd)

        ax_bot.plot(t, buffer.ttc_s, label="TTC (s)", linewidth=1.8, color="tab:red", zorder=3)
        ax_bot.axhline(1.5, color="gray", linestyle="--", linewidth=1.2, label="critical TTC = 1.5s")
        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylabel("TTC (s)")
        ax_bot.set_ylim(bottom=0)
        ax_bot.grid(True)
        ax_bot.legend(loc="upper right")
        _shade_hazard(ax_bot, t, buffer.hazard_brake_cmd)

        ax_text.text(
            0.0, 1.0,
            "Safety & Comfort Metrics:\n\n"
            "Top — Jerk (m/s³):\n"
            "• Rate of change of acceleration.\n"
            "• High jerk = passenger feels a sudden lurch.\n"
            "• Shaded = hazard brake active.\n"
            "• Smooth braking → jerk stays low.\n"
            "• Abrupt braking → sharp jerk spike.\n\n"
            "Bottom — Time-to-Collision (s):\n"
            "• TTC = ped_distance / ego_speed.\n"
            "• Only computed after trigger fires.\n"
            "• Below 1.5s (dashed line) = critical.\n"
            "• Lower TTC = more dangerous encounter.\n\n"
            "Use these together:\n"
            "Low TTC + high jerk = unsafe and uncomfortable.\n"
            "Low TTC + low jerk = risky but smooth braking.\n"
            "High TTC + high jerk = over-reaction (false alarm).",
            ha="left", va="top", fontsize=9
        )

    # If nothing was requested, tell you rather than silently doing nothing
    if not any([show_speed, show_braking_view, show_controls, show_lane_metrics,
                show_cone_params, show_safety_metrics]):
        print("plot_telemetry: no figures requested (all show_* flags are False).")
        return


    # -------------------------------------------------
    # Press 'q' in any figure window to close ALL plots
    # -------------------------------------------------
    def _close_all_on_q(event):
        if event.key == "q":
            plt.close("all")

    # Attach the key event to all currently open figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.canvas.mpl_connect("key_press_event", _close_all_on_q)

    plt.show()