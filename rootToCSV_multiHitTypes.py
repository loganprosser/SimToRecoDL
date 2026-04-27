import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

import numpy as np
import pandas as pd


INPUT_FILE = "/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_10mu_pt_0p5_50.root"
TREE_NAME = "trackingNtuple/tree"
OUTPUT_DIR = "outputCSVs/outputCSVs_hittype_compare"
SUMMARY_CSV = "rootToCSV_multiHitTypes_summary.csv"

ETACUT = 0.9
PTCUT = 1.9
MUON_ID = 13
SENTINEL = -999.0
DEFAULT_WORKERS = 32
DEFAULT_CHUNK_EVENTS = 25
DEFAULT_BATCH_SIZE = 1

REQUIRED_BRANCHES = [
    "sim_q",
    "sim_pt",
    "sim_pdgId",
    "sim_eta",
    "sim_simHitIdx",
    "sim_pca_pt",
    "sim_pca_eta",
    "sim_pca_phi",
    "sim_pca_dxy",
    "sim_pca_dz",
    "simhit_x",
    "simhit_y",
    "simhit_z",
    "simhit_hitType",
]

OPTIONAL_BRANCHES = [
    "simhit_isLower",
    "simhit_layer",
    "simhit_subdet",
]

HT4_BUCKETS = 6

VARIANTS = [
    {"name": "ht0_4bucket_first_seen", "ht0_buckets": 4, "policy": "first_seen"},
    {"name": "ht0_4bucket_min_r_absz_hitindex", "ht0_buckets": 4, "policy": "min_r_absz_hitindex"},
    {"name": "ht0_4bucket_min_absz_r_hitindex", "ht0_buckets": 4, "policy": "min_absz_r_hitindex"},
    {"name": "ht0_2bucket_first_seen", "ht0_buckets": 2, "policy": "first_seen"},
    {"name": "ht0_2bucket_min_r_absz_hitindex", "ht0_buckets": 2, "policy": "min_r_absz_hitindex"},
    {"name": "ht0_2bucket_min_absz_r_hitindex", "ht0_buckets": 2, "policy": "min_absz_r_hitindex"},
]


def import_uproot():
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit(
            "rootToCSV_multiHitTypes needs uproot to read ROOT files. Install uproot in this Python environment."
        ) from exc
    return uproot


def load_available_branches(tree):
    available = set(tree.keys())
    missing = [branch for branch in REQUIRED_BRANCHES if branch not in available]
    if missing:
        raise KeyError(f"Missing required branches: {missing}")
    return REQUIRED_BRANCHES + [branch for branch in OPTIONAL_BRANCHES if branch in available]


def value_at(data, branch, evt, index, default=-1):
    if branch not in data:
        return default
    values = data[branch][evt]
    if index >= len(values):
        return default
    return values[index]


def first_value(value, default=-1):
    try:
        if len(value) == 0:
            return default
        return value[0]
    except TypeError:
        return value


def selected_particle(data, evt, sim_idx, args):
    eta = data["sim_eta"][evt][sim_idx]
    pt = data["sim_pt"][evt][sim_idx]
    pid = data["sim_pdgId"][evt][sim_idx]
    q = data["sim_q"][evt][sim_idx]
    return abs(eta) <= args.eta_cut and pt >= args.pt_cut and abs(pid) == args.pdg_id and q != 0


def base_track_row(data, evt, sim_idx, event_number):
    q = data["sim_q"][evt][sim_idx]
    pca_pt = data["sim_pca_pt"][evt][sim_idx]
    return {
        "event": event_number,
        "sim_idx": sim_idx,
        "sim_q": q,
        "sim_pt": data["sim_pt"][evt][sim_idx],
        "sim_eta": data["sim_eta"][evt][sim_idx],
        "sim_pdgId": data["sim_pdgId"][evt][sim_idx],
        "pca_c": q / pca_pt,
        "pca_eta": data["sim_pca_eta"][evt][sim_idx],
        "pca_phi": data["sim_pca_phi"][evt][sim_idx],
        "pca_dxy": data["sim_pca_dxy"][evt][sim_idx],
        "pca_dz": data["sim_pca_dz"][evt][sim_idx],
    }


def hit_record(data, evt, sim_idx, hit_index, event_number):
    x = float(value_at(data, "simhit_x", evt, hit_index, np.nan))
    y = float(value_at(data, "simhit_y", evt, hit_index, np.nan))
    z = float(value_at(data, "simhit_z", evt, hit_index, np.nan))
    r = float(np.hypot(x, y))
    return {
        "event": event_number,
        "sim_idx": sim_idx,
        "hit_index": int(hit_index),
        "hit_type": int(first_value(value_at(data, "simhit_hitType", evt, hit_index, []), -1)),
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "phi": float(np.arctan2(y, x)),
        "layer": int(value_at(data, "simhit_layer", evt, hit_index, -1)),
        "subdet": int(value_at(data, "simhit_subdet", evt, hit_index, -1)),
        "is_lower": int(value_at(data, "simhit_isLower", evt, hit_index, -1)),
    }


def ht0_bucket(layer, n_buckets):
    if n_buckets == 4:
        return layer if 1 <= layer <= 4 else None
    if n_buckets == 2:
        if layer in (1, 2):
            return 1
        if layer in (3, 4):
            return 2
    return None


def ht4_bucket(layer):
    return layer if 1 <= layer <= HT4_BUCKETS else None


def better_hit(candidate, current, policy):
    if current is None:
        return True
    if policy == "first_seen":
        return False
    if policy == "min_r_absz_hitindex":
        return (candidate["r"], abs(candidate["z"]), candidate["hit_index"]) < (
            current["r"],
            abs(current["z"]),
            current["hit_index"],
        )
    if policy == "min_absz_r_hitindex":
        return (abs(candidate["z"]), candidate["r"], candidate["hit_index"]) < (
            abs(current["z"]),
            current["r"],
            current["hit_index"],
        )
    raise ValueError(f"Unknown policy: {policy}")


def empty_variant_columns(ht0_buckets):
    row = {}
    for bucket_count, prefix in [(ht0_buckets, "ht0"), (HT4_BUCKETS, "ht4")]:
        row[f"{prefix}_count"] = 0
        row[f"{prefix}_present"] = 0
        for slot in range(1, bucket_count + 1):
            for field in ["x", "y", "z", "r", "phi"]:
                row[f"{prefix}_bucket_{slot}_{field}"] = SENTINEL
            for field in ["layer", "subdet", "is_lower", "hit_index"]:
                row[f"{prefix}_bucket_{slot}_{field}"] = -1
            row[f"{prefix}_bucket_{slot}_mask"] = 0
    return row


def fill_variant_row(base_row, ht0_hits, ht4_hits, variant):
    row = dict(base_row)
    row.update(empty_variant_columns(variant["ht0_buckets"]))
    row["hit_type_combo"] = "0+4"
    row["has_all_requested_hit_types"] = 1
    row["all_requested_present"] = 1
    row["selection_policy"] = variant["policy"]
    row["ht0_bucket_scheme"] = variant["ht0_buckets"]
    row["ht0_count"] = len(ht0_hits)
    row["ht4_count"] = len(ht4_hits)
    row["ht0_present"] = 1
    row["ht4_present"] = 1

    ht0_slots = {}
    for hit in ht0_hits:
        bucket = ht0_bucket(hit["layer"], variant["ht0_buckets"])
        if bucket is None:
            continue
        current = ht0_slots.get(bucket)
        if better_hit(hit, current, variant["policy"]):
            ht0_slots[bucket] = hit

    ht4_slots = {}
    for hit in ht4_hits:
        bucket = ht4_bucket(hit["layer"])
        if bucket is None:
            continue
        current = ht4_slots.get(bucket)
        if better_hit(hit, current, variant["policy"]):
            ht4_slots[bucket] = hit

    for prefix, slots in [("ht0", ht0_slots), ("ht4", ht4_slots)]:
        for bucket, hit in sorted(slots.items()):
            for field in ["x", "y", "z", "r", "phi", "layer", "subdet", "is_lower", "hit_index"]:
                row[f"{prefix}_bucket_{bucket}_{field}"] = hit[field]
            row[f"{prefix}_bucket_{bucket}_mask"] = 1

    return row


def process_chunk(task):
    uproot = import_uproot()
    rows_by_variant = {variant["name"]: [] for variant in VARIANTS}
    coverage = Counter()

    with uproot.open(task["input"]) as root_file:
        tree = root_file[task["tree"]]
        for start in range(task["start"], task["stop"], task["batch_size"]):
            stop = min(start + task["batch_size"], task["stop"])
            data = tree.arrays(task["branches"], entry_start=start, entry_stop=stop, library="np")

            for local_evt, event_number in enumerate(range(start, stop)):
                coverage["events_processed"] += 1
                n_particles = len(data["sim_pdgId"][local_evt])
                coverage["sim_particles_seen"] += n_particles

                for sim_idx in range(n_particles):
                    if not selected_particle(data, local_evt, sim_idx, task["args"]):
                        coverage["sim_particles_rejected"] += 1
                        continue

                    coverage["selected_tracks"] += 1
                    base_row = base_track_row(data, local_evt, sim_idx, event_number)
                    ht0_hits = []
                    ht4_hits = []

                    for hit_index in data["sim_simHitIdx"][local_evt][sim_idx]:
                        hit = hit_record(data, local_evt, sim_idx, int(hit_index), event_number)
                        if not (np.isfinite(hit["x"]) and np.isfinite(hit["y"]) and np.isfinite(hit["z"])):
                            continue
                        if hit["hit_type"] == 0:
                            ht0_hits.append(hit)
                        elif hit["hit_type"] == 4:
                            if task["lower_only_ht4"] and hit["is_lower"] != 1:
                                continue
                            ht4_hits.append(hit)

                    if ht0_hits:
                        coverage["tracks_with_ht0"] += 1
                    if ht4_hits:
                        coverage["tracks_with_ht4"] += 1
                    if not (ht0_hits and ht4_hits):
                        continue

                    coverage["tracks_with_ht0_and_ht4"] += 1
                    for variant in VARIANTS:
                        rows_by_variant[variant["name"]].append(
                            fill_variant_row(base_row, ht0_hits, ht4_hits, variant)
                        )

    return {"rows_by_variant": rows_by_variant, "coverage": dict(coverage)}


def build_tasks(args, branches, n_events):
    tasks = []
    for start in range(0, n_events, args.chunk_events):
        stop = min(start + args.chunk_events, n_events)
        tasks.append(
            {
                "input": args.input,
                "tree": args.tree,
                "branches": branches,
                "start": start,
                "stop": stop,
                "batch_size": args.batch_size,
                "lower_only_ht4": args.lower_only_ht4,
                "args": args,
            }
        )
    return tasks


def build_summary(coverage, rows_by_variant):
    rows = [
        {"metric": "events_processed", "value": int(coverage["events_processed"])},
        {"metric": "sim_particles_seen", "value": int(coverage["sim_particles_seen"])},
        {"metric": "selected_tracks", "value": int(coverage["selected_tracks"])},
        {"metric": "sim_particles_rejected", "value": int(coverage["sim_particles_rejected"])},
        {"metric": "tracks_with_ht0", "value": int(coverage["tracks_with_ht0"])},
        {"metric": "tracks_with_ht4", "value": int(coverage["tracks_with_ht4"])},
        {"metric": "tracks_with_ht0_and_ht4", "value": int(coverage["tracks_with_ht0_and_ht4"])},
    ]
    for variant_name, rows_list in rows_by_variant.items():
        rows.append({"metric": f"rows_{variant_name}", "value": int(len(rows_list))})
    return pd.DataFrame(rows)


def build_csvs(args):
    uproot = import_uproot()
    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        branches = load_available_branches(tree)
        n_events = tree.num_entries if args.max_events == 0 else min(tree.num_entries, args.max_events)

    tasks = build_tasks(args, branches, n_events)
    if args.workers <= 1 or len(tasks) <= 1:
        results = [process_chunk(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_chunk, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    coverage = Counter()
    rows_by_variant = {variant["name"]: [] for variant in VARIANTS}
    for result in results:
        coverage.update(result["coverage"])
        for variant_name, rows_list in result["rows_by_variant"].items():
            rows_by_variant[variant_name].extend(rows_list)

    os.makedirs(args.output_dir, exist_ok=True)
    for variant in VARIANTS:
        variant_name = variant["name"]
        path = os.path.join(args.output_dir, f"{variant_name}.csv")
        pd.DataFrame(rows_by_variant[variant_name]).to_csv(path, index=False)
        print(f"Saved {len(rows_by_variant[variant_name]):,} rows to {path}")

    summary_path = os.path.join(args.output_dir, SUMMARY_CSV)
    build_summary(coverage, rows_by_variant).to_csv(summary_path, index=False)
    print(f"Processed {n_events:,} events using {args.workers} worker(s) across {len(tasks):,} chunk tasks.")
    print(f"Saved summary to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build concurrent hit-type comparison CSVs for tracks that have both hitType 0 and hitType 4."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Input ROOT file.")
    parser.add_argument("--tree", default=TREE_NAME, help="Tree path inside the ROOT file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--eta-cut", type=float, default=ETACUT)
    parser.add_argument("--pt-cut", type=float, default=PTCUT)
    parser.add_argument("--pdg-id", type=int, default=MUON_ID)
    parser.add_argument("--max-events", type=int, default=0, help="Events to process. Use 0 for all entries.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Events to read per uproot batch.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker processes.")
    parser.add_argument("--chunk-events", type=int, default=DEFAULT_CHUNK_EVENTS, help="Events assigned per worker task.")
    parser.add_argument(
        "--lower-only-ht4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require hitType 4 hits to come from lower sensors.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.chunk_events < 1:
        raise SystemExit("--chunk-events must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    return args


if __name__ == "__main__":
    build_csvs(parse_args())
