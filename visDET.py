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
OUTPUT_SUMMARY = "visDET_hit_type_summary.csv"

DEFAULT_MAX_EVENTS = 0
DEFAULT_CANDIDATES_PER_EVENT = 5000
DEFAULT_MAX_PER_HIT_TYPE = 350
DEFAULT_MAX_TOTAL_HITS = 2500
DEFAULT_INVENTORY_BATCH_SIZE = 20
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

# The repo only documents hitType 4 as silicon / OT. Keep every label numeric
# so unknown codes are explicit instead of looking like a decoded category.
HIT_TYPE_LABELS = {
    -1: "hitType -1: empty simhit_hitType",
    0: "hitType 0: undocumented",
    1: "hitType 1: undocumented",
    2: "hitType 2: undocumented",
    3: "hitType 3: undocumented",
    4: "hitType 4: silicon / OT",
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


def hit_type_label(hit_type):
    hit_type = int(hit_type)
    return HIT_TYPE_LABELS.get(hit_type, f"hitType {hit_type}: undocumented")


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


def import_uproot():
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("visDET needs uproot to read ROOT files. Install uproot in this Python environment.") from exc
    return uproot


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


def event_limit(total_entries, requested_events):
    if requested_events > 0:
        return min(total_entries, requested_events)
    return total_entries


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
        "hit_type_label": hit_type_label(hit_type),
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
    uproot = import_uproot()

    rng = np.random.default_rng(args.seed)
    records_by_type = defaultdict(list)
    seen_by_type = Counter()
    valid_seen_by_type = Counter()
    total_hits_scanned = 0

    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        branches = load_available_branches(tree)
        n_events = event_limit(tree.num_entries, args.max_events)

        for start in range(0, n_events, args.batch_size):
            stop = min(start + args.batch_size, n_events)
            batch = tree.arrays(branches, entry_start=start, entry_stop=stop, library="np")

            for local_event, event_number in enumerate(range(start, stop)):
                event_arrays = {branch: batch[branch][local_event] for branch in branches}
                n_hits = event_hit_count(event_arrays)
                if n_hits == 0:
                    continue

                if args.sample_all_hits:
                    hit_indices = range(n_hits)
                else:
                    n_candidates = min(n_hits, args.candidates_per_event)
                    hit_indices = rng.choice(n_hits, size=n_candidates, replace=False)

                for hit_index in hit_indices:
                    hit_type = int(scalar_at(event_arrays, "simhit_hitType", int(hit_index), -1))
                    seen_by_type[hit_type] += 1
                    total_hits_scanned += 1

                    record = make_hit_record(event_arrays, event_number, int(hit_index))
                    if not (np.isfinite(record["x"]) and np.isfinite(record["y"]) and np.isfinite(record["z"])):
                        continue

                    valid_seen_by_type[hit_type] += 1
                    if len(records_by_type[hit_type]) < args.max_per_hit_type:
                        records_by_type[hit_type].append(record)
                        continue

                    replacement_index = rng.integers(0, valid_seen_by_type[hit_type])
                    if replacement_index < args.max_per_hit_type:
                        records_by_type[hit_type][replacement_index] = record

    records = [record for hit_type in sorted(records_by_type) for record in records_by_type[hit_type]]
    df = pd.DataFrame(records)
    if len(df) > args.max_total_hits:
        df = balanced_downsample(df, args.max_total_hits, args.seed)

    return df, seen_by_type, valid_seen_by_type, total_hits_scanned, n_events


def scan_hit_type_inventory(args):
    uproot = import_uproot()
    counts = Counter()

    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        if "simhit_hitType" not in set(tree.keys()):
            raise KeyError("Missing required branch: simhit_hitType")

        n_events = event_limit(tree.num_entries, args.inventory_max_events)

        for start in range(0, n_events, args.inventory_batch_size):
            stop = min(start + args.inventory_batch_size, n_events)
            batch = tree.arrays(["simhit_hitType"], entry_start=start, entry_stop=stop, library="np")

            for local_event in range(stop - start):
                for hit_type_values in batch["simhit_hitType"][local_event]:
                    counts[int(first_value(hit_type_values, -1))] += 1

    return counts, n_events


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


def summarize_hit_types(df, seen_by_type, valid_seen_by_type=None, inventory_by_type=None):
    valid_seen_by_type = valid_seen_by_type or Counter()
    inventory_by_type = inventory_by_type or Counter()
    all_hit_types = sorted(set(inventory_by_type) | set(seen_by_type) | set(valid_seen_by_type))
    rows = []
    for hit_type in all_hit_types:
        type_df = df[df["hit_type"] == hit_type] if not df.empty else pd.DataFrame()
        subdet_counts = {}
        layer_counts = {}
        if not type_df.empty:
            subdet_counts = {
                f"{int(subdet)}:{SUBDET_LABELS.get(int(subdet), f'subdet {int(subdet)}')}": int(count)
                for subdet, count in type_df["subdet"].value_counts().sort_index().items()
            }
            layer_counts = {
                str(int(layer)): int(count)
                for layer, count in type_df["layer"].value_counts().sort_index().items()
            }

        rows.append(
            {
                "hit_type": int(hit_type),
                "hit_type_label": hit_type_label(hit_type),
                "inventory_hits_seen": int(inventory_by_type.get(hit_type, 0)),
                "sample_scan_hits_seen": int(seen_by_type[hit_type]),
                "sample_scan_valid_xyz_seen": int(valid_seen_by_type[hit_type]),
                "sampled_hits_saved": int(len(type_df)),
                "subdet_breakdown": "; ".join(f"{key}={value}" for key, value in subdet_counts.items()),
                "layer_breakdown": "; ".join(f"{key}={value}" for key, value in layer_counts.items()),
                "r_min": float(type_df["r"].min()) if not type_df.empty else np.nan,
                "r_median": float(type_df["r"].median()) if not type_df.empty else np.nan,
                "r_max": float(type_df["r"].max()) if not type_df.empty else np.nan,
                "z_min": float(type_df["z"].min()) if not type_df.empty else np.nan,
                "z_median": float(type_df["z"].median()) if not type_df.empty else np.nan,
                "z_max": float(type_df["z"].max()) if not type_df.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


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
        raise ValueError("No hits were collected; try increasing --max-events or checking valid x/y/z branches.")

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
        label = f"{hit_type_label(hit_type)} | {SUBDET_LABELS.get(int(subdet), f'subdet {int(subdet)}')}"
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
    count_labels = [hit_type_label(hit_type).replace(": ", "\n") for hit_type in counts.index]
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
        "Color = numeric hitType, marker = subdet, size ~= energy loss when available."
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
    parser.add_argument("--summary-name", default=OUTPUT_SUMMARY, help="Output hit-type summary CSV filename.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help="Event entries to inspect for plotting samples. Use 0 for all entries.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Events to read per uproot batch.")
    parser.add_argument(
        "--sample-all-hits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inspect every hit in the selected events and reservoir-sample each hitType for plotting.",
    )
    parser.add_argument(
        "--inventory-hit-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exactly scan simhit_hitType over the file before plotting.",
    )
    parser.add_argument(
        "--inventory-max-events",
        type=int,
        default=0,
        help="Events to scan for exact hitType inventory. Use 0 for all entries.",
    )
    parser.add_argument(
        "--inventory-batch-size",
        type=int,
        default=DEFAULT_INVENTORY_BATCH_SIZE,
        help="Events to read per exact hitType inventory batch.",
    )
    parser.add_argument(
        "--candidates-per-event",
        type=int,
        default=DEFAULT_CANDIDATES_PER_EVENT,
        help="Random hits inspected per event only when --no-sample-all-hits is used.",
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

    inventory_by_type = Counter()
    inventory_events = 0
    if args.inventory_hit_types:
        inventory_by_type, inventory_events = scan_hit_type_inventory(args)

    df, seen_by_type, valid_seen_by_type, total_hits_scanned, sample_events = collect_balanced_hits(args)

    csv_path = os.path.join(args.output_dir, args.csv_name)
    plot_path = os.path.join(args.output_dir, args.plot_name)
    summary_path = os.path.join(args.output_dir, args.summary_name)
    summary_df = summarize_hit_types(df, seen_by_type, valid_seen_by_type, inventory_by_type)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_detector(df, plot_path)

    if args.inventory_hit_types:
        print(f"Exact hitType inventory scanned {inventory_events:,} event entries.")
        for hit_type, count in sorted(inventory_by_type.items()):
            print(f"  {hit_type_label(hit_type)}: inventory={count:,}")
    else:
        print("Exact hitType inventory skipped; only sample-pass counts are available.")

    scan_mode = "all hits" if args.sample_all_hits else f"up to {args.candidates_per_event:,} random hits per event"
    print(f"Sample pass scanned {total_hits_scanned:,} hits across {sample_events:,} event entries ({scan_mode}).")
    print("Hit types plotted/saved:")
    for row in summary_df.itertuples(index=False):
        print(
            f"  {row.hit_type_label}: inventory={row.inventory_hits_seen:,}, "
            f"scan={row.sample_scan_hits_seen:,}, valid_xyz={row.sample_scan_valid_xyz_seen:,}, "
            f"saved={row.sampled_hits_saved:,}, subdets=[{row.subdet_breakdown}]"
        )
    print(f"Saved {len(df):,} sampled hits to {csv_path}")
    print(f"Saved hit-type summary to {summary_path}")
    print(f"Saved detector map to {plot_path}")


if __name__ == "__main__":
    main()
