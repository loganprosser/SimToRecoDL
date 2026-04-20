import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ====== CONFIG ======
INPUT_FILE = "/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_10mu_pt_0p5_50.root"
TREE_NAME = "trackingNtuple/tree"
OUTPUT_DIR = "detector_vis"
OUTPUT_CSV = "visDET_hits.csv"
OUTPUT_PLOT = "visDET_detector_map.png"

DEFAULT_MAX_EVENTS = 80
DEFAULT_CANDIDATES_PER_EVENT = 5000
DEFAULT_MAX_PER_HIT_TYPE = 350
DEFAULT_MAX_TOTAL_HITS = 2500
RANDOM_SEED = 13

REQUIRED_BRANCHES = ["simhit_x", "simhit_y", "simhit_z", "simhit_hitType"]
OPTIONAL_BRANCHES = [
    "simhit_isLower",
    "simhit_isUpper",
    "simhit_isStack",
    "simhit_subdet",
    "simhit_layer",
    "simhit_module",
    "simhit_moduleType",
    "simhit_process",
    "simhit_eloss",
    "simhit_tof",
    "simhit_simTrkIdx",
]

# Friendly names are intentionally conservative; unknown codes still plot fine.
HIT_TYPE_LABELS = {
    -1: "unknown",
    0: "type 0",
    1: "type 1",
    2: "type 2",
    3: "type 3",
    4: "silicon / OT",
}

SUBDET_LABELS = {
    0: "unknown",
    1: "pixel barrel",
    2: "pixel endcap",
    3: "strip inner barrel",
    4: "strip inner disks",
    5: "strip outer barrel",
    6: "strip endcap",
}

COLOR_CYCLE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
    "#b2df8a",
    "#fb9a99",
]

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]


def first_value(value, default=-1):
    """Return the first scalar from a scalar or small vector-like object."""
    try:
        if len(value) == 0:
            return default
        return value[0]
    except TypeError:
        return value


def value_at(event_arrays, branch, hit_index, default=np.nan):
    if branch not in event_arrays:
        return default
    values = event_arrays[branch]
    if hit_index >= len(values):
        return default
    return values[hit_index]


def scalar_at(event_arrays, branch, hit_index, default=np.nan):
    return first_value(value_at(event_arrays, branch, hit_index, default), default)


def load_available_branches(tree):
    available = set(tree.keys())
    missing = [branch for branch in REQUIRED_BRANCHES if branch not in available]
    if missing:
        raise KeyError(f"Missing required branches: {missing}")
    return REQUIRED_BRANCHES + [branch for branch in OPTIONAL_BRANCHES if branch in available]


def event_hit_count(event_arrays):
    counts = [len(event_arrays[branch]) for branch in REQUIRED_BRANCHES[:3]]
    counts.append(len(event_arrays["simhit_hitType"]))
    return min(counts)


def make_hit_record(event_arrays, event_number, hit_index):
    x = float(value_at(event_arrays, "simhit_x", hit_index))
    y = float(value_at(event_arrays, "simhit_y", hit_index))
    z = float(value_at(event_arrays, "simhit_z", hit_index))
    r = float(np.hypot(x, y))
    phi = float(np.arctan2(y, x))
    hit_type = int(scalar_at(event_arrays, "simhit_hitType", hit_index, -1))
    subdet = int(scalar_at(event_arrays, "simhit_subdet", hit_index, -1))
    layer = int(scalar_at(event_arrays, "simhit_layer", hit_index, -1))

    return {
        "event": event_number,
        "hit_index": hit_index,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "phi": phi,
        "hit_type": hit_type,
        "hit_type_label": HIT_TYPE_LABELS.get(hit_type, f"type {hit_type}"),
        "subdet": subdet,
        "subdet_label": SUBDET_LABELS.get(subdet, f"subdet {subdet}"),
        "layer": layer,
        "is_lower": int(scalar_at(event_arrays, "simhit_isLower", hit_index, -1)),
        "is_upper": int(scalar_at(event_arrays, "simhit_isUpper", hit_index, -1)),
        "is_stack": int(scalar_at(event_arrays, "simhit_isStack", hit_index, -1)),
        "module": int(scalar_at(event_arrays, "simhit_module", hit_index, -1)),
        "module_type": int(scalar_at(event_arrays, "simhit_moduleType", hit_index, -1)),
        "process": int(scalar_at(event_arrays, "simhit_process", hit_index, -1)),
        "eloss": float(scalar_at(event_arrays, "simhit_eloss", hit_index, np.nan)),
        "tof": float(scalar_at(event_arrays, "simhit_tof", hit_index, np.nan)),
        "sim_trk_idx": int(scalar_at(event_arrays, "simhit_simTrkIdx", hit_index, -1)),
    }


def collect_balanced_hits(args):
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("visDET needs uproot to read ROOT files. Install uproot in this Python environment.") from exc

    rng = np.random.default_rng(args.seed)
    records_by_type = defaultdict(list)
    seen_by_type = Counter()
    total_candidates = 0

    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        branches = load_available_branches(tree)
        n_events = min(tree.num_entries, args.max_events)

        for start in range(0, n_events, args.batch_size):
            stop = min(start + args.batch_size, n_events)
            batch = tree.arrays(branches, entry_start=start, entry_stop=stop, library="np")

            for local_event, event_number in enumerate(range(start, stop)):
                event_arrays = {branch: batch[branch][local_event] for branch in branches}
                n_hits = event_hit_count(event_arrays)
                if n_hits == 0:
                    continue

                n_candidates = min(n_hits, args.candidates_per_event)
                candidate_indices = rng.choice(n_hits, size=n_candidates, replace=False)
                total_candidates += n_candidates

                for hit_index in candidate_indices:
                    hit_type = int(scalar_at(event_arrays, "simhit_hitType", int(hit_index), -1))
                    seen_by_type[hit_type] += 1
                    if len(records_by_type[hit_type]) >= args.max_per_hit_type:
                        continue
                    record = make_hit_record(event_arrays, event_number, int(hit_index))
                    if np.isfinite(record["x"]) and np.isfinite(record["y"]) and np.isfinite(record["z"]):
                        records_by_type[hit_type].append(record)

    records = [record for hit_type in sorted(records_by_type) for record in records_by_type[hit_type]]
    df = pd.DataFrame(records)
    if len(df) > args.max_total_hits:
        df = balanced_downsample(df, args.max_total_hits, args.seed)

    return df, seen_by_type, total_candidates


def balanced_downsample(df, max_total_hits, seed):
    counts = df["hit_type"].value_counts().sort_index()
    n_types = len(counts)
    base_quota = max(1, max_total_hits // n_types)

    pieces = []
    leftovers = []
    for hit_type, count in counts.items():
        group = df[df["hit_type"] == hit_type]
        take = min(count, base_quota)
        pieces.append(group.sample(n=take, random_state=seed + int(hit_type) + 101))
        if count > take:
            leftovers.append(group.drop(pieces[-1].index))

    out = pd.concat(pieces, ignore_index=False)
    remaining = max_total_hits - len(out)
    if remaining > 0 and leftovers:
        leftover_df = pd.concat(leftovers, ignore_index=False)
        take = min(remaining, len(leftover_df))
        out = pd.concat(
            [out, leftover_df.sample(n=take, random_state=seed + 7919)],
            ignore_index=False,
        )

    return out.sort_values(["hit_type", "event", "hit_index"]).reset_index(drop=True)


def draw_detector_guides(ax_side, ax_xy, df):
    from matplotlib.patches import Circle

    if df.empty:
        return

    finite = df[np.isfinite(df["r"]) & np.isfinite(df["z"])]
    if finite.empty:
        return

    z_abs = float(np.nanpercentile(np.abs(finite["z"]), 99.5))
    z_lim = max(80.0, z_abs * 1.08)
    r_lim = max(40.0, float(np.nanpercentile(finite["r"], 99.5)) * 1.12)

    layer_radii = []
    grouped = finite[(finite["layer"] >= 0) & (finite["r"] > 0)].groupby(["subdet", "layer"])
    for (_, _), group in grouped:
        if len(group) >= 4:
            layer_radii.append(float(group["r"].median()))

    if len(layer_radii) < 4:
        layer_radii = [25, 35, 50, 68, 88, 110]

    layer_radii = sorted(set(round(radius, 1) for radius in layer_radii))
    for radius in layer_radii:
        if radius > r_lim * 1.05:
            continue
        ax_side.hlines(radius, -z_lim, z_lim, color="#c9ced6", linewidth=0.7, alpha=0.55, zorder=0)
        ax_side.hlines(-radius, -z_lim, z_lim, color="#c9ced6", linewidth=0.7, alpha=0.25, zorder=0)
        ax_xy.add_patch(
            Circle((0, 0), radius, fill=False, color="#c9ced6", linewidth=0.7, alpha=0.45, zorder=0)
        )

    for z_edge in [-z_lim * 0.62, z_lim * 0.62]:
        ax_side.vlines(z_edge, -r_lim, r_lim, color="#d9a441", linewidth=1.1, alpha=0.45, zorder=0)

    ax_side.axhline(0, color="#424852", linewidth=1.0, alpha=0.65, zorder=0)
    ax_side.axvline(0, color="#424852", linewidth=0.8, alpha=0.35, zorder=0)
    ax_xy.axhline(0, color="#424852", linewidth=0.8, alpha=0.35, zorder=0)
    ax_xy.axvline(0, color="#424852", linewidth=0.8, alpha=0.35, zorder=0)

    ax_side.set_xlim(-z_lim, z_lim)
    ax_side.set_ylim(-r_lim, r_lim)
    ax_xy.set_xlim(-r_lim, r_lim)
    ax_xy.set_ylim(-r_lim, r_lim)


def marker_sizes(df):
    if "eloss" not in df or df["eloss"].isna().all():
        return np.full(len(df), 24.0)
    eloss = df["eloss"].to_numpy(dtype=float)
    valid = np.isfinite(eloss) & (eloss > 0)
    sizes = np.full(len(df), 18.0)
    if valid.any():
        scaled = np.log1p(eloss[valid])
        lo, hi = np.nanpercentile(scaled, [5, 95])
        if hi > lo:
            sizes[valid] = 16 + 38 * np.clip((scaled - lo) / (hi - lo), 0, 1)
    return sizes


def plot_detector(df, output_path):
    plot_dir = os.path.dirname(output_path) or "."
    mpl_cache_dir = os.path.join(plot_dir, ".matplotlib_cache")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", mpl_cache_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "visDET needs matplotlib to draw the detector map. Install matplotlib in this Python environment."
        ) from exc

    if df.empty:
        raise ValueError("No hits were collected; try increasing --max-events or --candidates-per-event.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.38])
    ax_side = fig.add_subplot(grid[:, 0])
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_counts = fig.add_subplot(grid[1, 1])

    draw_detector_guides(ax_side, ax_xy, df)

    colors = {hit_type: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, hit_type in enumerate(sorted(df["hit_type"].unique()))}
    subdets = sorted(df["subdet"].dropna().unique())
    markers = {subdet: MARKERS[i % len(MARKERS)] for i, subdet in enumerate(subdets)}
    sizes = pd.Series(marker_sizes(df), index=df.index)

    for (hit_type, subdet), group in df.groupby(["hit_type", "subdet"], sort=True):
        label = f"{HIT_TYPE_LABELS.get(int(hit_type), f'type {int(hit_type)}')} | {SUBDET_LABELS.get(int(subdet), f'subdet {int(subdet)}')}"
        color = colors[hit_type]
        marker = markers[subdet]
        ax_side.scatter(
            group["z"],
            group["r"],
            s=sizes.loc[group.index],
            c=color,
            marker=marker,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            label=label,
        )
        ax_side.scatter(
            group["z"],
            -group["r"],
            s=sizes.loc[group.index] * 0.55,
            c=color,
            marker=marker,
            alpha=0.20,
            edgecolors="none",
        )
        ax_xy.scatter(
            group["x"],
            group["y"],
            s=sizes.loc[group.index] * 0.82,
            c=color,
            marker=marker,
            alpha=0.58,
            edgecolors="white",
            linewidths=0.3,
        )

    ax_side.set_title("visDET sampled sim hits: side view", fontsize=16, weight="bold")
    ax_side.set_xlabel("z [cm]")
    ax_side.set_ylabel("radius [cm] mirrored for detector shape")
    ax_side.grid(color="#e2e6eb", linewidth=0.8)

    ax_xy.set_title("end view", fontsize=13, weight="bold")
    ax_xy.set_xlabel("x [cm]")
    ax_xy.set_ylabel("y [cm]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(color="#e2e6eb", linewidth=0.8)

    counts = df["hit_type"].value_counts().sort_index()
    count_colors = [colors[hit_type] for hit_type in counts.index]
    count_labels = [HIT_TYPE_LABELS.get(int(hit_type), f"type {int(hit_type)}") for hit_type in counts.index]
    ax_counts.bar(count_labels, counts.values, color=count_colors, alpha=0.82)
    ax_counts.set_title("plotted hits by type", fontsize=12, weight="bold")
    ax_counts.set_ylabel("count")
    ax_counts.tick_params(axis="x", labelrotation=28)
    ax_counts.grid(axis="y", color="#e2e6eb", linewidth=0.8)
    ax_counts.grid(axis="x", visible=False)

    handles, labels = ax_side.get_legend_handles_labels()
    if handles:
        max_items = 12
        ax_side.legend(
            handles[:max_items],
            labels[:max_items],
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            framealpha=0.92,
            title="hit type | subdet",
            title_fontsize=9,
        )

    subtitle = (
        f"{len(df):,} plotted hits, capped per type for readability. "
        "Color = hit type, marker = subdet, size ~= energy loss when available."
    )
    fig.suptitle(subtitle, fontsize=11, y=1.01)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample varied sim hits from a tracking ROOT ntuple and draw detector-like maps."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Input ROOT file.")
    parser.add_argument("--tree", default=TREE_NAME, help="Tree path inside the ROOT file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for the CSV and plot.")
    parser.add_argument("--csv-name", default=OUTPUT_CSV, help="Output CSV filename.")
    parser.add_argument("--plot-name", default=OUTPUT_PLOT, help="Output plot filename.")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS, help="Maximum events to inspect.")
    parser.add_argument("--batch-size", type=int, default=4, help="Events to read per uproot batch.")
    parser.add_argument(
        "--candidates-per-event",
        type=int,
        default=DEFAULT_CANDIDATES_PER_EVENT,
        help="Random candidate hits inspected per event before balancing.",
    )
    parser.add_argument(
        "--max-per-hit-type",
        type=int,
        default=DEFAULT_MAX_PER_HIT_TYPE,
        help="Maximum collected hits for each simhit_hitType.",
    )
    parser.add_argument(
        "--max-total-hits",
        type=int,
        default=DEFAULT_MAX_TOTAL_HITS,
        help="Final maximum number of hits plotted.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, seen_by_type, total_candidates = collect_balanced_hits(args)

    csv_path = os.path.join(args.output_dir, args.csv_name)
    plot_path = os.path.join(args.output_dir, args.plot_name)
    df.to_csv(csv_path, index=False)
    plot_detector(df, plot_path)

    print(f"Inspected {total_candidates:,} candidate hits.")
    print("Candidate hit types seen:", dict(sorted(seen_by_type.items())))
    print(f"Saved {len(df):,} sampled hits to {csv_path}")
    print(f"Saved detector map to {plot_path}")


if __name__ == "__main__":
    main()
