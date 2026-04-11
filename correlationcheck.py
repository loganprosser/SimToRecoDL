import argparse
import os

import numpy as np
import pandas as pd

from helpers_data import DEFAULT_DATA_PATH, DEFAULT_TARGET_COLS, build_hit_feature_cols


"""
How to run:

Quick terminal-only sanity check:
    python3 SimToRecoDL/correlationcheck.py

Check only the solo pca_dxy target:
    python3 SimToRecoDL/correlationcheck.py --target-mode solo

Use Spearman rank correlation instead of Pearson:
    python3 SimToRecoDL/correlationcheck.py --method spearman

Save report, CSVs, and heatmap:
    python3 SimToRecoDL/correlationcheck.py --save

Save report and CSVs without a heatmap:
    python3 SimToRecoDL/correlationcheck.py --save --no-heatmap
"""


DEFAULT_OUTPUT_DIR = "correlation_checks"


def get_default_target_cols(target_mode):
    if target_mode == "solo":
        return ["pca_dxy"]
    return DEFAULT_TARGET_COLS


def safe_corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan

    x = x[valid]
    y = y[valid]

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def compute_feature_target_corr(df, feature_cols, target_cols, method):
    cols = feature_cols + target_cols
    if method in ("pearson", "spearman"):
        corr = df[cols].corr(method=method)
        return corr.loc[feature_cols, target_cols]

    rows = {}
    for feature in feature_cols:
        rows[feature] = {}
        for target in target_cols:
            rows[feature][target] = safe_corr(
                df[feature].to_numpy(dtype=np.float64),
                df[target].to_numpy(dtype=np.float64),
            )
    return pd.DataFrame.from_dict(rows, orient="index")[target_cols]


def summarize_top_correlations(corr, top_n):
    lines = []
    for target in corr.columns:
        ranked = corr[target].dropna().reindex(
            corr[target].dropna().abs().sort_values(ascending=False).index
        )

        lines.append(f"\nTop {top_n} absolute correlations for {target}:")
        if ranked.empty:
            lines.append("  No finite correlations found.")
            continue

        for feature, value in ranked.head(top_n).items():
            lines.append(f"  {feature:16s} corr = {value: .6f} | abs = {abs(value): .6f}")

    return "\n".join(lines)


def summarize_feature_groups(corr):
    feature_groups = {
        "x": [idx for idx in corr.index if idx.endswith("_x")],
        "y": [idx for idx in corr.index if idx.endswith("_y")],
        "z": [idx for idx in corr.index if idx.endswith("_z")],
        "r": [idx for idx in corr.index if idx.endswith("_r")],
        "mask": [idx for idx in corr.index if idx.endswith("_mask")],
    }

    rows = []
    for group_name, features in feature_groups.items():
        if not features:
            continue
        group_corr = corr.loc[features].abs()
        row = {"feature_group": group_name}
        for target in corr.columns:
            row[target] = group_corr[target].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def build_report_text(corr, group_summary, method, csv_path, top_n):
    lines = [
        "FEATURE/TARGET CORRELATION CHECK",
        "=" * 80,
        f"csv_path: {csv_path}",
        f"method: {method}",
        f"n_features: {len(corr.index)}",
        f"n_targets: {len(corr.columns)}",
        "",
        "Interpretation:",
        "  Large absolute values suggest a simple linear/rank relationship.",
        "  Small values do not prove the target is unlearnable; nonlinear networks can use",
        "  feature combinations that pairwise correlation does not see.",
        summarize_top_correlations(corr, top_n),
        "",
        "Mean absolute correlation by feature group:",
    ]

    if group_summary.empty:
        lines.append("  No grouped correlations found.")
    else:
        lines.append(group_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return "\n".join(lines)


def write_report(path, corr, group_summary, method, csv_path, top_n):
    report_text = build_report_text(corr, group_summary, method, csv_path, top_n)

    with open(path, "w") as f:
        f.write(report_text)
        f.write("\n")


def make_heatmap(corr, save_path):
    import matplotlib.pyplot as plt

    fig_width = max(8, len(corr.columns) * 2.0)
    fig_height = max(8, len(corr.index) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(corr.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature vs target correlation")
    fig.colorbar(im, ax=ax, label="Correlation")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check pairwise correlations between track hit inputs and target outputs."
    )
    parser.add_argument("--csv-path", default=DEFAULT_DATA_PATH, help="Path to filtered_particles.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports/CSVs/plots")
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "manual_pearson"])
    parser.add_argument("--target-mode", default="full", choices=["full", "solo"])
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--save", action="store_true", help="Save report, CSVs, and heatmap to output-dir")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap png generation when --save is on")
    return parser.parse_args()


def main():
    args = parse_args()

    feature_cols = build_hit_feature_cols(n_layers=6)
    target_cols = get_default_target_cols(args.target_mode)

    df = pd.read_csv(args.csv_path)
    df = df[feature_cols + target_cols].copy()
    df[feature_cols] = df[feature_cols].replace(-999.0, 0.0)

    corr = compute_feature_target_corr(df, feature_cols, target_cols, args.method)
    abs_corr = corr.abs()
    group_summary = summarize_feature_groups(corr)

    report_text = build_report_text(
        corr=corr,
        group_summary=group_summary,
        method=args.method,
        csv_path=args.csv_path,
        top_n=args.top_n,
    )

    print(report_text)

    if not args.save:
        print("\nNot saving files. Re-run with --save to write CSVs, report, and heatmap.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    prefix = f"{args.target_mode}_{args.method}"
    corr_path = os.path.join(args.output_dir, f"{prefix}_feature_target_correlation.csv")
    abs_corr_path = os.path.join(args.output_dir, f"{prefix}_feature_target_abs_correlation.csv")
    group_path = os.path.join(args.output_dir, f"{prefix}_feature_group_abs_correlation.csv")
    report_path = os.path.join(args.output_dir, f"{prefix}_correlation_report.txt")
    heatmap_path = os.path.join(args.output_dir, f"{prefix}_feature_target_correlation_heatmap.png")

    corr.to_csv(corr_path)
    abs_corr.to_csv(abs_corr_path)
    group_summary.to_csv(group_path, index=False)
    write_report(report_path, corr, group_summary, args.method, args.csv_path, args.top_n)

    if not args.no_heatmap:
        make_heatmap(corr, heatmap_path)

    print("Saved correlation check outputs:")
    print(f"  report: {report_path}")
    print(f"  correlation csv: {corr_path}")
    print(f"  abs correlation csv: {abs_corr_path}")
    print(f"  group summary csv: {group_path}")
    if not args.no_heatmap:
        print(f"  heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
