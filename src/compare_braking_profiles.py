"""
compare_braking_profiles.py

Reads a sweep_summary.csv produced by sweep.py and generates comparison
figures across braking profiles.

Usage:
    python compare_braking_profiles.py
    python compare_braking_profiles.py runs/20260401_120000/sweep_summary.csv

If no path is given, the script finds the most recent sweep folder automatically.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# =====================================================================
# Config
# =====================================================================

# Display order + colors for the four profiles
PROFILE_ORDER = ["proportional_ramp", "step_constant", "cautious_ramp", "exponential"]
PROFILE_COLORS = {
    "proportional_ramp": "#4C72B0",   # blue
    "step_constant":     "#DD8452",   # orange
    "cautious_ramp":     "#55A868",   # green
    "exponential":       "#C44E52",   # red
}
PROFILE_LABELS = {
    "proportional_ramp": "Proportional\nRamp (baseline)",
    "step_constant":     "Step\nConstant",
    "cautious_ramp":     "Cautious\nRamp",
    "exponential":       "Exponential",
}

OUTCOME_COLORS = {
    "full_stop":      "#2ca02c",   # green
    "slowed_avoided": "#ff7f0e",   # orange
    "collision":      "#d62728",   # red
}

# Text shown in the key panel on every figure describing the four profiles.
_PROFILE_KEY_TEXT = (
    "Braking Profiles:  "
    "Prop. Ramp = brake force increases gradually as the obstacle gets closer (baseline);  "
    "Step Constant = full braking force applied immediately the moment a hazard is detected;  "
    "Cautious Ramp = same gradual increase as Prop. Ramp but capped at 50% max force;  "
    "Exponential = very light braking at first, rapidly increasing as the obstacle closes in."
)

# Figure-specific context lines added below the shared profile key.
_FIG_CONTEXT = {
    "boxplots": (
        "HOW TO READ A BOX PLOT: Colored box = middle 50% of runs (box bottom = 25th percentile, box top = 75th percentile).  "
        "White line inside box = median (half of runs above, half below).  "
        "Thin whisker lines above/below box = full range of typical runs.  Dots beyond whiskers = outlier runs.\n"
        "Jerk panels (left 2): non-collision runs only — collision runs excluded since they represent failed stops, not normal braking comfort.\n"
        "TTC and Min Ped Distance (right 2): all runs included.  TTC varies across runs because it equals distance ÷ ego speed — the 3 test speeds (25/35/45 mph) naturally produce different TTC values.\n"
        "Min Ped Distance is 3D Euclidean center-to-center distance — NOT a collision threshold. A pedestrian safely passing alongside the ego (lateral pass) can produce a small distance while being completely safe. "
        "Collision ground truth is CARLA's physics sensor (see fig3). Individual dots are colored by actual outcome — use dot color, not distance value, to determine whether a close approach was a collision or a safe pass."
    ),
    "scatter": (
        "Large dot = profile mean across all non-collision runs.  Shaded ellipse = 1.5σ spread (how consistent the profile is).\n"
        "X-axis = ride roughness (lower = smoother).  Y-axis = closest the car got to the pedestrian (higher = safer).\n"
        "✕ markers = collision runs (excluded from ellipses).  Min distance for collisions is ~1–2 m, not 0 — CARLA measures center-to-center; hitbox contact occurs before centers meet.\n"
        "Top-left is ideal: smooth AND safe.  Tighter ellipse = more consistent profile across scenarios."
    ),
    "outcomes": (
        "Each bar = % of runs ending in that outcome for a given braking profile.  n = number of runs.\n"
        "Full Stop (green) = car fully stopped before reaching the pedestrian — best outcome.\n"
        "Slowed & Avoided (orange) = car slowed enough to avoid collision but did not fully stop.\n"
        "Collision (red) = car physically struck the pedestrian."
    ),
    "jerk_speed": (
        "Bar height = average jerk (ride roughness) across successful (non-collision) runs at that speed, grouped by braking profile.\n"
        "Collision runs are excluded — their jerk reflects a failed stop, not the profile's comfort under normal operation.\n"
        "Higher bar = bumpier, more uncomfortable braking on average.  Shows whether comfort penalty scales with speed."
    ),
}


def _add_key(fig, context_key: str):
    """
    Add a structured key panel at the bottom of a figure.
    Row 1: color-coded profile chips with short descriptions.
    Row 2: figure-specific "how to read" note.
    """
    # Reserve space at the bottom
    fig.subplots_adjust(bottom=0.36)

    # Add a dedicated axes for the key (below the plot area)
    ax_key = fig.add_axes([0.01, 0.01, 0.98, 0.30])
    ax_key.set_axis_off()

    # Background box
    ax_key.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        facecolor="#f5f5f5",
        edgecolor="#cccccc",
        linewidth=0.8,
        transform=ax_key.transAxes,
        clip_on=False,
    ))

    # --- Row 1: "Braking Profiles:" header alone, then chips on next line ---
    ax_key.text(
        0.01, 0.97, "Braking Profiles:",
        transform=ax_key.transAxes,
        fontsize=8, fontweight="bold", va="top", color="#222222",
    )

    # Profile chips evenly spread across full width, below the header
    profile_descs = [
        ("proportional_ramp", "Prop. Ramp",    "brake force increases gradually\nas obstacle closes in (baseline)"),
        ("step_constant",     "Step Constant", "full brake applied immediately\non any hazard detection"),
        ("cautious_ramp",     "Cautious Ramp", "same as Prop. Ramp but\ncapped at 50% max force"),
        ("exponential",       "Exponential",   "very gentle at first, then ramps\nup sharply as obstacle closes in"),
    ]

    x_positions = [0.125, 0.375, 0.625, 0.875]  # evenly spaced across full width

    for (profile, short_name, desc), cx in zip(profile_descs, x_positions):
        color = PROFILE_COLORS[profile]

        # Colored name badge — sits on the line below the header
        ax_key.text(
            cx, 0.78, f"■ {short_name}",
            transform=ax_key.transAxes,
            fontsize=8.5, fontweight="bold", va="top", ha="center",
            color=color,
        )
        # Description below badge — pre-broken with \n, no wrap=True
        ax_key.text(
            cx, 0.52, desc,
            transform=ax_key.transAxes,
            fontsize=7.2, va="top", ha="center", color="#444444",
        )

    # Divider line
    ax_key.axhline(0.30, color="#cccccc", linewidth=0.8, xmin=0.0, xmax=1.0)

    # --- Row 2: "How to read" header on its own line, context text below it ---
    ax_key.text(
        0.01, 0.26, "How to read this chart:",
        transform=ax_key.transAxes,
        fontsize=7.5, fontweight="bold", va="top", color="#222222",
    )
    ax_key.text(
        0.01, 0.14, _FIG_CONTEXT[context_key],
        transform=ax_key.transAxes,
        fontsize=7.2, va="top", color="#444444",
    )



# =====================================================================
# Helpers
# =====================================================================

def _draw_confidence_ellipse(ax, x_data, y_data, color, n_std=1.5):
    """
    Draw a covariance ellipse (n_std standard deviations) around the mean
    of x_data, y_data. Shows both the center and spread of a profile's runs.
    Requires at least 3 points to compute a meaningful covariance.
    """
    if len(x_data) < 3:
        return
    cov = np.cov(x_data, y_data)
    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(vals[0], 0))
    height = 2 * n_std * np.sqrt(max(vals[1], 0))
    ellipse = mpatches.Ellipse(
        xy=(mean_x, mean_y),
        width=width, height=height, angle=angle,
        facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.5, linestyle="--",
        zorder=2,
    )
    ax.add_patch(ellipse)

def _find_latest_csv() -> str:
    """Return path to sweep_summary.csv in the most recently created sweep folder."""
    candidates = glob.glob(os.path.join("runs", "*", "sweep_summary.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No sweep_summary.csv found under runs/. "
            "Run sweep.py first, or pass the CSV path as an argument."
        )
    # sort by folder modification time
    candidates.sort(key=lambda p: os.path.getmtime(os.path.dirname(p)), reverse=True)
    return candidates[0]


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Coerce numeric columns (some may be empty strings for failed runs)
    numeric_cols = [
        "min_ped_distance_m", "min_ttc_s", "max_jerk_mps3", "mean_jerk_mps3",
        "time_to_stop_s", "ego_speed_at_trigger_mph", "trigger_ttc_s",
        "braking_ramp_up_per_s", "brake_headway_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop crashed runs
    crashed = df["error"].notna() & (df["error"] == "CRASHED")
    n_crashed = crashed.sum()
    if n_crashed:
        print(f"[compare] Dropping {n_crashed} crashed run(s) from plots.")
        df = df[~crashed].copy()

    # Keep only known profiles so ordering works
    df = df[df["brake_profile"].isin(PROFILE_ORDER)].copy()
    df["brake_profile"] = pd.Categorical(df["brake_profile"], categories=PROFILE_ORDER, ordered=True)

    return df


# =====================================================================
# Plot 1: Box plots — key metrics per profile
# =====================================================================

def plot_metric_boxplots(df: pd.DataFrame, out_dir: str):
    """Four side-by-side box plots: mean_jerk, max_jerk, min_ttc, min_ped_dist.

    Jerk panels use only non-collision runs — collision runs represent failed
    stops and skew comfort metrics. Safety panels (TTC, ped distance) use all
    runs since collisions are the most informative data points there.
    """
    df_safe = df[df["outcome"] != "collision"].copy()

    metrics = [
        ("mean_jerk_mps3",    "Mean Jerk (m/s³)",     "lower = smoother ride\n(non-collision runs only)",        df_safe),
        ("max_jerk_mps3",     "Peak Jerk (m/s³)",     "lower = less harsh braking\n(non-collision runs only)",   df_safe),
        ("min_ttc_s",         "Min TTC (s)",           "higher = more safety margin\n(all runs, incl. collisions)",  df),
        ("min_ped_distance_m","Min Ped Distance (m)",  "higher = more clearance\n(dots colored by actual outcome)",  df),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Braking Profile Comparison — Key Metrics", fontsize=14, fontweight="bold")

    for ax, (col, ylabel, subtitle, source_df) in zip(axes, metrics):
        data_by_profile = [
            source_df.loc[source_df["brake_profile"] == p, col].dropna().to_numpy(dtype=float)
            for p in PROFILE_ORDER
        ]
        colors = [PROFILE_COLORS[p] for p in PROFILE_ORDER]

        bp = ax.boxplot(
            data_by_profile,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        short_labels = ["Prop.\nRamp", "Step\nConst.", "Cautious\nRamp", "Exp."]
        ax.set_xticks(range(1, len(PROFILE_ORDER) + 1))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(subtitle, fontsize=8, color="#555555")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Add anatomy guide to the first panel only
        if col == "mean_jerk_mps3":
            ax.text(
                0.98, 0.98,
                "Box = middle 50% of runs\n"
                "━ white line = median\n"
                "│ whiskers = full range\n"
                "• dots = outliers",
                transform=ax.transAxes,
                fontsize=6, color="#555555", va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85, pad=3, linewidth=0.7),
            )

        # On the min_ped_distance panel: overlay individual run dots colored by
        # outcome so collision runs are unmistakably visible in red.
        # Without this, collision runs appear as tiny outlier dots that are
        # easy to miss, making the figure look contradictory with fig3.
        if col == "min_ped_distance_m":
            _outcome_dot_colors = {
                "collision":      "#d62728",
                "full_stop":      "#2ca02c",
                "slowed_avoided": "#ff7f0e",
            }
            rng = np.random.default_rng(seed=0)   # fixed seed so jitter is reproducible
            for j, profile in enumerate(PROFILE_ORDER, start=1):
                sub_p = source_df[source_df["brake_profile"] == profile].dropna(subset=[col])
                jitter = rng.uniform(-0.18, 0.18, size=len(sub_p))
                for (_, row), jit in zip(sub_p.iterrows(), jitter):
                    dot_color = _outcome_dot_colors.get(str(row["outcome"]), "#888888")
                    is_collision = row["outcome"] == "collision"
                    ax.scatter(
                        j + jit, row[col],
                        c=dot_color,
                        s=55 if is_collision else 30,
                        alpha=0.95 if is_collision else 0.6,
                        zorder=5,
                        edgecolors="white" if is_collision else "none",
                        linewidths=0.8,
                        marker="D" if is_collision else "o",   # diamond = collision, circle = safe
                    )

            # NOTE: No collision threshold line is drawn here.
            # min_ped_distance is 3D Euclidean center-to-center distance, which
            # does NOT reliably predict collisions. A pedestrian alongside the ego
            # (lateral safe pass) can produce 2m Euclidean distance while being
            # completely safe; a head-on glancing collision can register at 2.6m.
            # Collision ground truth comes from CARLA's physics sensor (fig3),
            # not from a distance threshold. Individual run outcomes are shown
            # as colored dots instead — those are the authoritative collision labels.

            # Mini legend for the dot colors — placed inside the panel
            legend_elements = [
                plt.scatter([], [], marker="D", c="#d62728", s=55, edgecolors="white", linewidths=0.8, label="Collision"),
                plt.scatter([], [], marker="o", c="#2ca02c", s=30, label="Full stop"),
                plt.scatter([], [], marker="o", c="#ff7f0e", s=30, label="Slowed avoided"),
            ]
            ax.legend(handles=legend_elements, fontsize=6, loc="upper right",
                      title="Run outcome", title_fontsize=6, framealpha=0.85)

    _add_key(fig, "boxplots")
    plt.tight_layout(rect=(0, 0.34, 1, 1))
    path = os.path.join(out_dir, "fig1_metric_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[compare] Saved -> {path}")
    plt.show()


# =====================================================================
# Plot 2: Safety vs Comfort scatter
# =====================================================================

def plot_safety_vs_comfort(df: pd.DataFrame, out_dir: str):
    """
    Safety vs comfort: profile means as large markers + covariance ellipses.
    Collision runs shown separately as X markers but excluded from ellipses.
    This replaces the original per-run scatter, which had overlapping dots
    that made it impossible to distinguish profile clusters.
    """
    df_safe = df[df["outcome"] != "collision"].copy()
    df_coll = df[df["outcome"] == "collision"].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    for profile in PROFILE_ORDER:
        sub = df_safe[df_safe["brake_profile"] == profile].dropna(
            subset=["mean_jerk_mps3", "min_ped_distance_m"]
        )
        if len(sub) == 0:
            continue
        color  = PROFILE_COLORS[profile]
        x_vals = sub["mean_jerk_mps3"].to_numpy(dtype=float)
        y_vals = sub["min_ped_distance_m"].to_numpy(dtype=float)

        # Covariance ellipse showing 1.5σ spread of this profile's runs
        _draw_confidence_ellipse(ax, x_vals, y_vals, color, n_std=1.5)

        # Mean marker — large, high zorder so it sits on top of the ellipse
        ax.scatter(
            x_vals.mean(), y_vals.mean(),
            c=color, s=220, zorder=5,
            edgecolors="white", linewidths=1.5,
            label=profile.replace("_", " "),
        )

        # Label next to the mean point
        short = PROFILE_LABELS[profile].replace("\n", " ")
        ax.annotate(
            short,
            (x_vals.mean(), y_vals.mean()),
            textcoords="offset points", xytext=(9, 4),
            fontsize=7.5, color=color, fontweight="bold", zorder=6,
        )

    # Collision runs as X markers (excluded from ellipses but shown for transparency)
    if len(df_coll) > 0:
        ax.scatter(
            df_coll["mean_jerk_mps3"],
            df_coll["min_ped_distance_m"],
            marker="x", c="#d62728", s=90, linewidths=2,
            zorder=6, label="Collision run (excluded from ellipses)",
        )

    ax.set_xlabel("Mean Jerk (m/s³)  →  higher = rougher ride", fontsize=10)
    ax.set_ylabel("Min Pedestrian Distance (m)  →  higher = safer", fontsize=10)
    ax.set_title("Safety vs. Comfort Trade-off by Braking Profile", fontsize=12, fontweight="bold")

    ax.text(
        0.02, 0.98, "← ideal region\n(smooth + safe)",
        transform=ax.transAxes, fontsize=7.5, color="#333333", va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5),
    )

    ax.legend(title="Brake profile", fontsize=8.5, loc="lower right")
    ax.grid(linestyle="--", alpha=0.2)

    _add_key(fig, "scatter")
    plt.tight_layout(rect=(0, 0.34, 1, 1))
    path = os.path.join(out_dir, "fig2_safety_vs_comfort.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[compare] Saved -> {path}")
    plt.show()


# =====================================================================
# Plot 3: Outcome stacked bar chart per profile
# =====================================================================

def plot_outcome_bars(df: pd.DataFrame, out_dir: str):
    """
    Stacked bar: for each profile, proportion of full_stop / slowed_avoided / collision.
    """
    outcome_order = ["full_stop", "slowed_avoided", "collision"]

    counts = (
        df.groupby(["brake_profile", "outcome"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=outcome_order, fill_value=0)
    )
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = np.zeros(len(PROFILE_ORDER))
    short_labels = ["Prop. Ramp", "Step Constant", "Cautious Ramp", "Exponential"]

    for outcome in outcome_order:
        vals = [percents.loc[p, outcome] if p in percents.index else 0.0 for p in PROFILE_ORDER]
        bars = ax.bar(
            short_labels,
            vals, # pyright: ignore
            bottom=bottom,
            color=OUTCOME_COLORS[outcome],
            label=outcome.replace("_", " ").title(),
            edgecolor="white",
            linewidth=0.5,
        )
        # Label segments: inside if >= 8%, or annotate above bar if smaller
        # but only for collision (so every non-zero collision % is always visible).
        for bar, val, bot in zip(bars, vals, bottom):
            val = float(val) if val == val else 0.0  # type: ignore[arg-type]
            if val <= 0.0:
                continue
            x_center = bar.get_x() + bar.get_width() / 2
            if val >= 8.0:
                # Label fits inside the bar segment
                ax.text(
                    x_center, bot + val / 2,  # pyright: ignore
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
            elif outcome == "collision":
                # Sliver too small for inside label — annotate above bar with arrow
                ax.annotate(
                    f"{val:.0f}%",
                    xy=(x_center, bot + val),  # pyright: ignore
                    xytext=(x_center, bot + val + 6),  # pyright: ignore
                    ha="center", va="bottom",
                    fontsize=8, color=OUTCOME_COLORS["collision"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=OUTCOME_COLORS["collision"],
                                   lw=0.8),
                )
        bottom += np.array(vals)

    ax.set_ylabel("Percentage of Runs (%)", fontsize=10)
    ax.set_title("Scenario Outcomes by Braking Profile", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Raw counts above each bar (clear of the stacked segments)
    totals = [counts.loc[p].sum() if p in counts.index else 0 for p in PROFILE_ORDER]
    for i, total in enumerate(totals):
        ax.text(i, 102, f"n={total}", ha="center", va="bottom", fontsize=8, color="#555555")

    _add_key(fig, "outcomes")
    plt.tight_layout(rect=(0, 0.34, 1, 1))
    path = os.path.join(out_dir, "fig3_outcome_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[compare] Saved -> {path}")
    plt.show()


# =====================================================================
# Plot 4: Mean jerk by speed + profile (grouped bar)
# =====================================================================

def plot_jerk_by_speed(df: pd.DataFrame, out_dir: str):
    """
    Grouped bar: mean_jerk per profile, grouped by ego speed.
    Collision runs excluded — their jerk reflects a failed stop, not normal operation.
    Shows how comfort penalty of each profile scales with speed.
    """
    df = df[df["outcome"] != "collision"].copy()
    speeds = sorted(df["target_mph"].dropna().unique())
    n_speeds = len(speeds)
    n_profiles = len(PROFILE_ORDER)
    bar_width = 0.18
    x = np.arange(n_speeds)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, profile in enumerate(PROFILE_ORDER):
        means = []
        for speed in speeds:
            vals = df.loc[
                (df["brake_profile"] == profile) & (df["target_mph"] == speed),
                "mean_jerk_mps3"
            ].dropna() # pyright: ignore
            means.append(vals.mean() if len(vals) > 0 else 0.0)

        offset = (i - n_profiles / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            color=PROFILE_COLORS[profile],
            label=profile.replace("_", " "),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0f} mph" for s in speeds], fontsize=10)
    ax.set_ylabel("Mean Jerk (m/s³)", fontsize=10)
    ax.set_title("Comfort (Mean Jerk) by Speed and Braking Profile", fontsize=12, fontweight="bold")
    ax.legend(title="Brake profile", fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    _add_key(fig, "jerk_speed")
    plt.tight_layout(rect=(0, 0.34, 1, 1))
    path = os.path.join(out_dir, "fig4_jerk_by_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[compare] Saved -> {path}")
    plt.show()


# =====================================================================
# Summary table printed to console
# =====================================================================

def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PROFILE SUMMARY (means across all runs)")
    print("=" * 70)
    metrics = ["mean_jerk_mps3", "max_jerk_mps3", "min_ttc_s", "min_ped_distance_m", "time_to_stop_s"]
    agg = df.groupby("brake_profile", observed=True)[metrics].mean()
    agg = agg.reindex(PROFILE_ORDER)
    print(agg.round(2).to_string())

    print("\n" + "=" * 70)
    print("OUTCOME COUNTS PER PROFILE")
    print("=" * 70)
    counts = df.groupby(["brake_profile", "outcome"], observed=True).size().unstack(fill_value=0)
    print(counts.reindex(PROFILE_ORDER).to_string())
    print()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    # Parse arguments:
    #   python compare_braking_profiles.py                          <- latest sweep, all headways pooled
    #   python compare_braking_profiles.py path/to/sweep_summary.csv
    #   python compare_braking_profiles.py --headway 2.5           <- filter to baseline detection only
    #   python compare_braking_profiles.py --headway 5.0           <- filter to early detection only
    #   python compare_braking_profiles.py path/to/csv --headway 2.5
    args = sys.argv[1:]
    csv_path   = None
    headway_filter = None   # None = use all runs

    i = 0
    while i < len(args):
        if args[i] == "--headway" and i + 1 < len(args):
            headway_filter = float(args[i + 1])
            i += 2
        elif not args[i].startswith("--"):
            csv_path = args[i]
            i += 1
        else:
            i += 1

    if csv_path is None:
        csv_path = _find_latest_csv()

    print(f"[compare] Loading: {csv_path}")
    df = _load(csv_path)

    # --- Headway filter ---
    if headway_filter is not None:
        if "brake_headway_s" not in df.columns:
            print(f"[compare] WARNING: brake_headway_s column not found — ignoring --headway filter.")
        else:
            before = len(df)
            df = df[df["brake_headway_s"] == headway_filter].copy()
            print(f"[compare] Filtered to brake_headway_s={headway_filter}s "
                  f"({len(df)} of {before} runs kept)")
            if df.empty:
                available = sorted(df["brake_headway_s"].unique()) if "brake_headway_s" in df.columns else []
                raise SystemExit(
                    f"No runs found with brake_headway_s={headway_filter}. "
                    f"Available values in this CSV: {available}"
                )

    headway_vals = (
        sorted(df["brake_headway_s"].dropna().unique().tolist())
        if "brake_headway_s" in df.columns else []
    )
    headway_label = (
        f"headway={headway_filter}s" if headway_filter is not None
        else ("all headways pooled" if len(headway_vals) > 1 else f"headway={headway_vals[0]}s")
    )

    print(f"[compare] {len(df)} valid runs | {df['brake_profile'].nunique()} profiles | {headway_label}\n")

    # Output figures alongside the CSV, with headway suffix when filtered
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if headway_filter is not None:
        suffix = f"_headway{headway_filter:.1f}s"
        # Patch figure save paths by monkey-patching plt.savefig via a subdir
        out_dir = os.path.join(out_dir, f"headway_{headway_filter:.1f}s")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[compare] Figures will be saved to subfolder: {out_dir}")

    print_summary_table(df)

    plot_metric_boxplots(df, out_dir)
    plot_safety_vs_comfort(df, out_dir)
    plot_outcome_bars(df, out_dir)
    plot_jerk_by_speed(df, out_dir)

    print(f"\n[compare] All figures saved to: {out_dir}")
