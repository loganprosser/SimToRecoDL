import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DEFAULT_DATA_PATH = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"

def build_hit_feature_cols(n_layers=6):
    feature_cols = []
    for j in range(1, n_layers + 1):
        feature_cols += [
            f"hit_{j}_x",
            f"hit_{j}_y",
            f"hit_{j}_z",
            f"hit_{j}_r",
            f"hit_{j}_mask",
        ]
    return feature_cols

def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def main():
    csv_path = DEFAULT_DATA_PATH
    n_layers = 6
    sentinel_value = -999.0
    sentinel_replacement = 0.0
    target_col = "pca_dxy"

    feature_cols = build_hit_feature_cols(n_layers)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    needed_cols = feature_cols + [target_col]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # raw features and target
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy(dtype=np.float64)

    # replace sentinel values
    X = X.replace(sentinel_value, sentinel_replacement)

    print("\n===== RAW pca_dxy stats =====")
    print(f"count : {len(y)}")
    print(f"mean  : {np.mean(y):.10f}")
    print(f"std   : {np.std(y):.10f}")
    print(f"min   : {np.min(y):.10f}")
    print(f"max   : {np.max(y):.10f}")
    print(f"mean(|y|): {np.mean(np.abs(y)):.10f}")
    print(f"median(|y|): {np.median(np.abs(y)):.10f}")

    near_zero_thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    print("\n===== Fraction near zero =====")
    for thr in near_zero_thresholds:
        frac = np.mean(np.abs(y) < thr)
        print(f"|pca_dxy| < {thr:.0e}: {frac:.6f}")

    print("\n===== Per-feature correlation with raw pca_dxy =====")
    corr_rows = []
    abs_y = np.abs(y)

    for col in feature_cols:
        x_col = X[col].to_numpy(dtype=np.float64)
        corr_y = safe_corr(x_col, y)
        corr_abs_y = safe_corr(x_col, abs_y)
        corr_rows.append((col, corr_y, abs(corr_y) if not np.isnan(corr_y) else np.nan, corr_abs_y))

    corr_df = pd.DataFrame(
        corr_rows,
        columns=["feature", "corr_with_dxy", "abs_corr_with_dxy", "corr_with_abs_dxy"]
    )

    corr_df_sorted = corr_df.sort_values("abs_corr_with_dxy", ascending=False)

    print("\nTop 20 by |corr(feature, pca_dxy)|")
    print(corr_df_sorted.head(20).to_string(index=False))

    abs_corr_absy_df = corr_df.copy()
    abs_corr_absy_df["abs_corr_with_abs_dxy"] = abs_corr_absy_df["corr_with_abs_dxy"].abs()
    abs_corr_absy_df = abs_corr_absy_df.sort_values("abs_corr_with_abs_dxy", ascending=False)

    print("\nTop 20 by |corr(feature, abs(pca_dxy))|")
    print(abs_corr_absy_df[["feature", "corr_with_abs_dxy", "abs_corr_with_abs_dxy"]].head(20).to_string(index=False))

    print("\n===== Simple linear baseline for learnability =====")
    X_np = X.to_numpy(dtype=np.float64)

    n = len(X_np)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = int(0.2 * n)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X_np[train_idx]
    X_val = X_np[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    baseline_zero_pred = np.zeros_like(y_val)
    ss_res_zero = np.sum((y_val - baseline_zero_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    zero_r2 = 1.0 - ss_res_zero / ss_tot if ss_tot > 0 else np.nan

    print(f"LinearRegression train R^2: {train_r2:.6f}")
    print(f"LinearRegression val   R^2: {val_r2:.6f}")
    print(f'Zero-predictor val R^2     : {zero_r2:.6f}')

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False)

    print("\nTop 20 linear-model coefficients by magnitude")
    print(coef_df.head(20).to_string(index=False))

    print("\n===== Quick interpretation guide =====")
    print("1. If almost all |corr| values are tiny, the inputs may not contain much direct signal for pca_dxy.")
    print("2. If linear val R^2 is near 0 or negative, a simple model cannot learn much from these features.")
    print("3. That does NOT prove a neural net cannot learn anything, but it does mean the mapping is weak or nonlinear.")
    print("4. If val R^2 is meaningfully positive, there is signal and the issue is more likely training setup / target handling / architecture.")

if __name__ == "__main__":
    main()