import argparse
import os
import sys
from typing import Iterable, List

import numpy as np
import pandas as pd


def parse_feature_list(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    features = [item.strip() for item in raw.split(",") if item.strip()]
    return features or None


def infer_numeric_features(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> List[str]:
    ref_numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
    cur_numeric = set(current_df.select_dtypes(include=[np.number]).columns)
    common = sorted(ref_numeric & cur_numeric)
    return [feature for feature in common if feature != "is_fraud"]


def build_bin_edges(reference_series: pd.Series, bins: int) -> np.ndarray | None:
    clean = reference_series.dropna().to_numpy()
    if clean.size == 0:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(clean, quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_psi_for_feature(
    reference_series: pd.Series,
    current_series: pd.Series,
    bins: int,
    epsilon: float = 1e-6,
) -> float | None:
    edges = build_bin_edges(reference_series, bins)
    if edges is None:
        return None

    ref_bucket = pd.cut(reference_series, bins=edges, include_lowest=True)
    cur_bucket = pd.cut(current_series, bins=edges, include_lowest=True)

    ref_dist = ref_bucket.value_counts(sort=False, normalize=True)
    cur_dist = cur_bucket.value_counts(sort=False, normalize=True)

    ref_dist = ref_dist.fillna(0.0).astype(float)
    cur_dist = cur_dist.fillna(0.0).astype(float)

    ref_dist = np.clip(ref_dist.to_numpy(), epsilon, None)
    cur_dist = np.clip(cur_dist.to_numpy(), epsilon, None)

    psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    return float(psi)


def classify_psi(psi_value: float | None, warn_threshold: float, alert_threshold: float) -> str:
    if psi_value is None or np.isnan(psi_value):
        return "no_data"
    if psi_value >= alert_threshold:
        return "alert"
    if psi_value >= warn_threshold:
        return "warn"
    return "ok"


def ensure_dir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_monitoring(
    reference_csv: str,
    current_csv: str,
    output_csv: str,
    features: Iterable[str] | None,
    bins: int,
    warn_threshold: float,
    alert_threshold: float,
) -> pd.DataFrame:
    reference_df = pd.read_csv(reference_csv)
    current_df = pd.read_csv(current_csv)

    if features is None:
        selected_features = infer_numeric_features(reference_df, current_df)
    else:
        selected_features = [f for f in features if f in reference_df.columns and f in current_df.columns]

    rows = []
    for feature in selected_features:
        ref_series = reference_df[feature]
        cur_series = current_df[feature]

        psi_value = compute_psi_for_feature(ref_series, cur_series, bins=bins)
        status = classify_psi(psi_value, warn_threshold, alert_threshold)

        rows.append(
            {
                "feature": feature,
                "psi": psi_value,
                "status": status,
                "reference_missing_rate": float(ref_series.isna().mean()),
                "current_missing_rate": float(cur_series.isna().mean()),
                "reference_mean": float(ref_series.mean()) if pd.api.types.is_numeric_dtype(ref_series) else np.nan,
                "current_mean": float(cur_series.mean()) if pd.api.types.is_numeric_dtype(cur_series) else np.nan,
                "reference_rows": int(ref_series.shape[0]),
                "current_rows": int(cur_series.shape[0]),
            }
        )

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(by="psi", ascending=False, na_position="last").reset_index(drop=True)

    ensure_dir_for_file(output_csv)
    report_df.to_csv(output_csv, index=False)
    return report_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PSI threshold monitoring for fraud features")
    parser.add_argument("--reference-csv", required=True, help="Path to reference CSV")
    parser.add_argument("--current-csv", required=True, help="Path to current batch CSV")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature list. If omitted, common numeric features are used.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of PSI bins")
    parser.add_argument("--psi-warn", type=float, default=0.10, help="Warning threshold")
    parser.add_argument("--psi-alert", type=float, default=0.25, help="Alert threshold")
    parser.add_argument(
        "--output-csv",
        default="monitoring/psi_report.csv",
        help="Path to output report CSV",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 2 if any feature is in alert state",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.bins < 2:
        print("--bins must be >= 2", file=sys.stderr)
        return 1

    if args.psi_warn > args.psi_alert:
        print("--psi-warn cannot be greater than --psi-alert", file=sys.stderr)
        return 1

    features = parse_feature_list(args.features)
    report_df = run_monitoring(
        reference_csv=args.reference_csv,
        current_csv=args.current_csv,
        output_csv=args.output_csv,
        features=features,
        bins=args.bins,
        warn_threshold=args.psi_warn,
        alert_threshold=args.psi_alert,
    )

    if report_df.empty:
        print("No matching features found. Report generated but empty.")
        return 0

    status_counts = report_df["status"].value_counts().to_dict()
    print(f"PSI report written to: {args.output_csv}")
    print(f"Status summary: {status_counts}")
    print("Top 5 features by PSI:")
    print(report_df[["feature", "psi", "status"]].head(5).to_string(index=False))

    if args.fail_on_alert and (report_df["status"] == "alert").any():
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
