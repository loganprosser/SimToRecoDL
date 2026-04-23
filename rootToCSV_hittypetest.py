import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

import numpy as np
import pandas as pd


INPUT_FILE = "/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_10mu_pt_0p5_50.root"
TREE_NAME = "trackingNtuple/tree"
OUTPUT_DIR = "outputCSVs_para"
MASKED_CSV = "filtered_particles_multihit_masked.csv"
COMPLETE_CSV = "filtered_particles_multihit_complete_0_4.csv"
LONG_CSV = "filtered_particles_multihit_long.csv"
SUMMARY_CSV = "filtered_particles_multihit_summary.csv"
COVERAGE_CSV = "filtered_particles_multihit_coverage.csv"
DEFAULT_WORKERS = 32
DEFAULT_CHUNK_EVENTS = 25

ETACUT = 0.9
PTCUT = 1.9
MUON_ID = 13
SENTINEL = -999.0

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
    "simhit_isUpper",
    "simhit_isStack",
    "simhit_subdet",
    "simhit_layer",
    "simhit_module",
    "simhit_moduleType",
]


def first_value(value, default=-1):
    try:
        if len(value) == 0:
            return default
        return value[0]
    except TypeError:
        return value


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


def parse_int_set(text):
    if text is None or text.strip() == "":
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_slots(text, hit_types, default_slots):
    slots = {hit_type: default_slots for hit_type in hit_types}
    if not text:
        return slots
    for part in text.split(","):
        if not part.strip():
            continue
        hit_type_text, count_text = part.split(":")
        slots[int(hit_type_text.strip())] = int(count_text.strip())
    return slots


def selected_particle(data, evt, sim_idx, args):
    eta = data["sim_eta"][evt][sim_idx]
    pt = data["sim_pt"][evt][sim_idx]
    pid = data["sim_pdgId"][evt][sim_idx]
    q = data["sim_q"][evt][sim_idx]
    return abs(eta) <= args.eta_cut and pt >= args.pt_cut and abs(pid) == args.pdg_id and q != 0


def rejection_reasons(data, evt, sim_idx, args):
    reasons = []
    eta = data["sim_eta"][evt][sim_idx]
    pt = data["sim_pt"][evt][sim_idx]
    pid = data["sim_pdgId"][evt][sim_idx]
    q = data["sim_q"][evt][sim_idx]
    if abs(eta) > args.eta_cut:
        reasons.append("eta")
    if pt < args.pt_cut:
        reasons.append("pt")
    if abs(pid) != args.pdg_id:
        reasons.append("pdg_id")
    if q == 0:
        reasons.append("zero_charge")
    return reasons


def base_track_row(data, evt, sim_idx):
    q = data["sim_q"][evt][sim_idx]
    pca_pt = data["sim_pca_pt"][evt][sim_idx]
    return {
        "event": evt,
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


def hit_record(data, evt, sim_idx, hit_index):
    x = float(value_at(data, "simhit_x", evt, hit_index, np.nan))
    y = float(value_at(data, "simhit_y", evt, hit_index, np.nan))
    z = float(value_at(data, "simhit_z", evt, hit_index, np.nan))
    hit_type = int(first_value(value_at(data, "simhit_hitType", evt, hit_index, []), -1))
    subdet = int(value_at(data, "simhit_subdet", evt, hit_index, -1))
    layer = int(value_at(data, "simhit_layer", evt, hit_index, -1))
    return {
        "event": evt,
        "sim_idx": sim_idx,
        "hit_index": int(hit_index),
        "hit_type": hit_type,
        "x": x,
        "y": y,
        "z": z,
        "r": float(np.hypot(x, y)),
        "phi": float(np.arctan2(y, x)),
        "subdet": subdet,
        "layer": layer,
        "is_lower": int(value_at(data, "simhit_isLower", evt, hit_index, -1)),
        "is_upper": int(value_at(data, "simhit_isUpper", evt, hit_index, -1)),
        "is_stack": int(value_at(data, "simhit_isStack", evt, hit_index, -1)),
        "module": int(value_at(data, "simhit_module", evt, hit_index, -1)),
        "module_type": int(value_at(data, "simhit_moduleType", evt, hit_index, -1)),
    }


def empty_slot_columns(hit_types, slots_by_type):
    columns = {}
    for hit_type in hit_types:
        prefix = f"ht{hit_type}"
        columns[f"{prefix}_count"] = 0
        columns[f"{prefix}_valid_count"] = 0
        columns[f"{prefix}_has_any"] = 0
        columns[f"{prefix}_present"] = 0
        columns[f"{prefix}_stored_count"] = 0
        columns[f"{prefix}_overflow_count"] = 0
        for slot in range(1, slots_by_type[hit_type] + 1):
            for field in ["x", "y", "z", "r", "phi"]:
                columns[f"{prefix}_hit_{slot}_{field}"] = SENTINEL
            for field in ["subdet", "layer", "is_lower", "is_upper", "is_stack", "module", "module_type", "hit_index"]:
                columns[f"{prefix}_hit_{slot}_{field}"] = -1
            columns[f"{prefix}_hit_{slot}_mask"] = 0
    return columns


def add_hits_to_wide_row(row, hits_by_type, hit_types, slots_by_type):
    for hit_type in hit_types:
        prefix = f"ht{hit_type}"
        hits = sorted(hits_by_type.get(hit_type, []), key=lambda item: (item["r"], abs(item["z"]), item["hit_index"]))
        row[f"{prefix}_count"] = len(hits)
        row[f"{prefix}_has_any"] = int(len(hits) > 0)
        row[f"{prefix}_present"] = int(len(hits) > 0)
        row[f"{prefix}_stored_count"] = 0
        row[f"{prefix}_overflow_count"] = 0

        slotted_hits = choose_slot_hits(hits, slots_by_type[hit_type])
        row[f"{prefix}_valid_count"] = len(slotted_hits)
        row[f"{prefix}_stored_count"] = len(slotted_hits)
        row[f"{prefix}_overflow_count"] = max(0, len(hits) - len(slotted_hits))

        for slot, hit in slotted_hits.items():
            for field in ["x", "y", "z", "r", "phi"]:
                row[f"{prefix}_hit_{slot}_{field}"] = hit[field]
            for field in ["subdet", "layer", "is_lower", "is_upper", "is_stack", "module", "module_type", "hit_index"]:
                row[f"{prefix}_hit_{slot}_{field}"] = hit[field]
            row[f"{prefix}_hit_{slot}_mask"] = 1


def choose_slot_hits(hits, n_slots):
    slotted_hits = {}
    for hit in hits:
        slot = int(hit["layer"])
        if 1 <= slot <= n_slots:
            existing = slotted_hits.get(slot)
            if existing is None or (hit["r"], abs(hit["z"]), hit["hit_index"]) < (
                existing["r"],
                abs(existing["z"]),
                existing["hit_index"],
            ):
                slotted_hits[slot] = hit
    return dict(sorted(slotted_hits.items()))


def process_event_chunk(task):
    uproot = import_uproot()
    hit_types = task["hit_types"]
    lower_only_hit_types = set(task["lower_only_hit_types"])
    slots_by_type = task["slots_by_type"]

    wide_rows = []
    long_rows = []
    track_type_combos = Counter()
    hit_type_counts = Counter()
    tracks_with_type = Counter()
    selected_hit_multiplicity = {hit_type: [] for hit_type in hit_types}
    coverage = Counter()
    rejection_counts = Counter()

    with uproot.open(task["input"]) as root_file:
        tree = root_file[task["tree"]]
        branches = task["branches"]

        for start in range(task["start"], task["stop"], task["batch_size"]):
            stop = min(start + task["batch_size"], task["stop"])
            data = tree.arrays(branches, entry_start=start, entry_stop=stop, library="np")

            for local_evt, evt in enumerate(range(start, stop)):
                n_particles = len(data["sim_pdgId"][local_evt])
                coverage["events_processed"] += 1
                coverage["sim_particles_seen"] += n_particles
                for sim_idx in range(n_particles):
                    reasons = rejection_reasons(data, local_evt, sim_idx, task["args"])
                    if reasons:
                        coverage["sim_particles_rejected"] += 1
                        for reason in reasons:
                            rejection_counts[reason] += 1
                        continue
                    coverage["selected_tracks"] += 1

                    row = base_track_row(data, local_evt, sim_idx)
                    row["event"] = evt
                    row.update(empty_slot_columns(hit_types, slots_by_type))

                    hits_by_type = {hit_type: [] for hit_type in hit_types}
                    for hit_index in data["sim_simHitIdx"][local_evt][sim_idx]:
                        record = hit_record(data, local_evt, sim_idx, int(hit_index))
                        hit_type = record["hit_type"]
                        if hit_type not in hits_by_type:
                            continue
                        if hit_type in lower_only_hit_types and record["is_lower"] != 1:
                            continue
                        if not (np.isfinite(record["x"]) and np.isfinite(record["y"]) and np.isfinite(record["z"])):
                            continue

                        record["event"] = evt
                        hits_by_type[hit_type].append(record)
                        hit_type_counts[hit_type] += 1
                        if task["write_long"]:
                            long_row = dict(row)
                            long_row.update(record)
                            long_rows.append(long_row)

                    present_types = tuple(hit_type for hit_type in hit_types if hits_by_type[hit_type])
                    track_type_combos[present_types] += 1
                    for hit_type in present_types:
                        tracks_with_type[hit_type] += 1
                    for hit_type in hit_types:
                        selected_hit_multiplicity[hit_type].append(len(hits_by_type[hit_type]))
                    row["hit_type_combo"] = "+".join(str(hit_type) for hit_type in present_types) if present_types else "none"
                    row["has_all_requested_hit_types"] = int(len(present_types) == len(hit_types))
                    row["all_requested_present"] = row["has_all_requested_hit_types"]
                    add_hits_to_wide_row(row, hits_by_type, hit_types, slots_by_type)
                    wide_rows.append(row)

    return {
        "wide_rows": wide_rows,
        "long_rows": long_rows,
        "track_type_combos": dict(track_type_combos),
        "hit_type_counts": dict(hit_type_counts),
        "tracks_with_type": dict(tracks_with_type),
        "selected_hit_multiplicity": {str(hit_type): values for hit_type, values in selected_hit_multiplicity.items()},
        "coverage": dict(coverage),
        "rejection_counts": dict(rejection_counts),
    }


def merge_counter_rows(chunks, key_field, value_field):
    counter = Counter()
    for chunk in chunks:
        for row in chunk:
            counter[row[key_field]] += row[value_field]
    return pd.DataFrame([{key_field: key, value_field: value} for key, value in sorted(counter.items())])


def merge_simple_counters(chunks):
    counter = Counter()
    for chunk in chunks:
        counter.update(chunk)
    return counter


def normalize_combo_key(key):
    if isinstance(key, tuple):
        return tuple(int(value) for value in key)
    if key in ("", None):
        return tuple()
    return tuple(int(value) for value in str(key).split(",") if str(value) != "")


def build_coverage_df(coverage, rejection_counts, tracks_with_type, selected_hit_multiplicity, track_type_combos, hit_types):
    rows = [
        {"metric": "events_processed", "value": int(coverage["events_processed"])},
        {"metric": "sim_particles_seen", "value": int(coverage["sim_particles_seen"])},
        {"metric": "selected_tracks", "value": int(coverage["selected_tracks"])},
        {"metric": "sim_particles_rejected", "value": int(coverage["sim_particles_rejected"])},
        {"metric": "complete_tracks_all_requested_hit_types", "value": int(track_type_combos.get(tuple(hit_types), 0))},
        {"metric": "masked_tracks_all_selected", "value": int(coverage["selected_tracks"])},
    ]
    selected_count = max(int(coverage["selected_tracks"]), 1)
    for reason, count in sorted(rejection_counts.items()):
        rows.append({"metric": f"rejected_by_{reason}", "value": int(count)})
    for hit_type in hit_types:
        counts = np.asarray(selected_hit_multiplicity.get(hit_type, []), dtype=float)
        present = int(tracks_with_type[hit_type])
        rows.extend(
            [
                {"metric": f"tracks_with_ht{hit_type}", "value": present},
                {"metric": f"tracks_without_ht{hit_type}", "value": int(coverage["selected_tracks"] - present)},
                {"metric": f"fraction_tracks_with_ht{hit_type}", "value": present / selected_count},
                {
                    "metric": f"mean_attached_ht{hit_type}_hits_per_selected_track",
                    "value": float(counts.mean()) if counts.size else 0.0,
                },
                {
                    "metric": f"median_attached_ht{hit_type}_hits_per_selected_track",
                    "value": float(np.median(counts)) if counts.size else 0.0,
                },
            ]
        )
    for combo, count in sorted(track_type_combos.items(), key=lambda item: (str(item[0]), item[1])):
        label = "+".join(str(hit_type) for hit_type in combo) if combo else "none"
        rows.append({"metric": f"tracks_combo_{label}", "value": int(count)})
        rows.append({"metric": f"fraction_tracks_combo_{label}", "value": int(count) / selected_count})
    return pd.DataFrame(rows)


def build_tasks(args, branches, n_events):
    hit_types = sorted(parse_int_set(args.hit_types))
    lower_only_hit_types = sorted(parse_int_set(args.lower_only_hit_types))
    slots_by_type = parse_slots(args.slots, hit_types, args.default_slots)
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
                "hit_types": hit_types,
                "lower_only_hit_types": lower_only_hit_types,
                "slots_by_type": slots_by_type,
                "args": args,
                "write_long": args.write_long,
            }
        )
    return tasks


def build_csvs(args):
    uproot = import_uproot()
    hit_types = sorted(parse_int_set(args.hit_types))
    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        branches = load_available_branches(tree)
        n_events = tree.num_entries if args.max_events == 0 else min(tree.num_entries, args.max_events)

    tasks = build_tasks(args, branches, n_events)
    if args.workers <= 1 or len(tasks) <= 1:
        results = [process_event_chunk(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_event_chunk, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    wide_df = pd.DataFrame([row for result in results for row in result["wide_rows"]])
    long_df = pd.DataFrame([row for result in results for row in result["long_rows"]])
    complete_df = (
        wide_df[wide_df["has_all_requested_hit_types"] == 1].copy()
        if "has_all_requested_hit_types" in wide_df
        else pd.DataFrame()
    )
    track_type_combos = Counter()
    hit_type_counts = Counter()
    tracks_with_type = Counter()
    coverage = Counter()
    rejection_counts = Counter()
    selected_hit_multiplicity = {hit_type: [] for hit_type in hit_types}

    for result in results:
        track_type_combos.update({normalize_combo_key(key): value for key, value in result["track_type_combos"].items()})
        hit_type_counts.update(result["hit_type_counts"])
        tracks_with_type.update(result["tracks_with_type"])
        coverage.update(result["coverage"])
        rejection_counts.update(result["rejection_counts"])
        for hit_type_str, values in result["selected_hit_multiplicity"].items():
            selected_hit_multiplicity[int(hit_type_str)].extend(values)

    summary_df = pd.DataFrame(
        [
            {
                "hit_type_combo": "+".join(str(hit_type) for hit_type in combo) if combo else "none",
                "track_count": count,
            }
            for combo, count in sorted(track_type_combos.items(), key=lambda item: (str(item[0]), item[1]))
        ]
    )
    coverage_df = build_coverage_df(
        coverage,
        rejection_counts,
        tracks_with_type,
        selected_hit_multiplicity,
        track_type_combos,
        hit_types,
    )
    hit_count_df = pd.DataFrame(
        [{"hit_type": hit_type, "attached_hit_count": count} for hit_type, count in sorted(hit_type_counts.items())]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    masked_path = os.path.join(args.output_dir, args.masked_csv)
    complete_path = os.path.join(args.output_dir, args.complete_csv)
    long_path = os.path.join(args.output_dir, args.long_csv)
    summary_path = os.path.join(args.output_dir, args.summary_csv)
    coverage_path = os.path.join(args.output_dir, args.coverage_csv)
    hit_count_path = os.path.join(args.output_dir, args.summary_csv.replace(".csv", "_hit_counts.csv"))

    wide_df.to_csv(masked_path, index=False)
    complete_df.to_csv(complete_path, index=False)
    if args.write_long:
        long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    hit_count_df.to_csv(hit_count_path, index=False)

    print(f"Selected {coverage['selected_tracks']:,} tracks from {coverage['sim_particles_seen']:,} sim particles.")
    print(f"Processed {n_events:,} events using {args.workers} worker(s) across {len(tasks):,} chunk tasks.")
    print(f"Saved {len(wide_df):,} masked/all-selected track rows to {masked_path}")
    print(f"Saved {len(complete_df):,} complete all-hit-type track rows to {complete_path}")
    if args.write_long:
        print(f"Saved {len(long_df):,} attached hit rows to {long_path}")
    print(f"Saved hit-type combo summary to {summary_path}")
    print(f"Saved coverage summary to {coverage_path}")
    print(f"Saved attached hit-count summary to {hit_count_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build track-level CSVs with multiple simhit_hitType groups attached to each selected sim particle."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Input ROOT file.")
    parser.add_argument("--tree", default=TREE_NAME, help="Tree path inside the ROOT file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--masked-csv", default=MASKED_CSV, help="Masked/all-selected one-row-per-track CSV.")
    parser.add_argument("--complete-csv", default=COMPLETE_CSV, help="Complete-only one-row-per-track CSV.")
    parser.add_argument("--long-csv", default=LONG_CSV, help="One-row-per-attached-hit CSV.")
    parser.add_argument("--summary-csv", default=SUMMARY_CSV, help="Track hit-type combo summary CSV.")
    parser.add_argument("--coverage-csv", default=COVERAGE_CSV, help="Selection and hit availability coverage CSV.")
    parser.add_argument("--hit-types", default="0,4", help="Comma-separated simhit_hitType values to attach per track.")
    parser.add_argument("--lower-only-hit-types", default="4", help="Hit types where only lower sensors are kept.")
    parser.add_argument("--default-slots", type=int, default=8, help="Default number of sorted hit slots per hit type.")
    parser.add_argument("--slots", default="0:12,4:6", help="Per-hitType slot overrides, e.g. 0:12,4:6.")
    parser.add_argument("--eta-cut", type=float, default=ETACUT)
    parser.add_argument("--pt-cut", type=float, default=PTCUT)
    parser.add_argument("--pdg-id", type=int, default=MUON_ID)
    parser.add_argument("--max-events", type=int, default=0, help="Events to process. Use 0 for all entries.")
    parser.add_argument("--batch-size", type=int, default=1, help="Events to read per uproot batch.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker processes.")
    parser.add_argument("--chunk-events", type=int, default=DEFAULT_CHUNK_EVENTS, help="Events assigned per worker task.")
    parser.add_argument("--write-long", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not parse_int_set(args.hit_types):
        raise SystemExit("--hit-types must include at least one integer hitType.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.chunk_events < 1:
        raise SystemExit("--chunk-events must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    return args


if __name__ == "__main__":
    build_csvs(parse_args())
