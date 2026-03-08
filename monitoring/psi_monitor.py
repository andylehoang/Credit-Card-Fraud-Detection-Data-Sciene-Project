import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_value: str) -> Path:
    path_obj = Path(path_value)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)


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


def find_csv_file(dataset_dir: Path, preferred_name: str) -> Path:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {dataset_dir}")

    preferred_lower = preferred_name.lower()
    exact = [f for f in csv_files if f.name.lower() == preferred_lower]
    if exact:
        return exact[0]

    return csv_files[0]


def resolve_input_paths(
    reference_csv: str | None,
    current_csv: str | None,
    dataset_dir: str | None,
    auto_kagglehub: bool,
    kaggle_dataset: str,
    data_dir: str,
) -> tuple[str, str]:
    if reference_csv and current_csv:
        resolved_ref = resolve_project_path(reference_csv)
        resolved_cur = resolve_project_path(current_csv)

        if resolved_ref.exists() and resolved_cur.exists():
            return str(resolved_ref), str(resolved_cur)

        if not auto_kagglehub and not dataset_dir:
            return str(resolved_ref), str(resolved_cur)

    resolved_dataset_dir: Path | None = None

    if auto_kagglehub:
        try:
            import kagglehub  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub is not installed. Install it with: pip install kagglehub"
            ) from exc

        download_path = kagglehub.dataset_download(kaggle_dataset)
        resolved_dataset_dir = Path(download_path)
        print(f"Downloaded dataset path: {resolved_dataset_dir}")
    elif dataset_dir:
        resolved_dataset_dir = resolve_project_path(dataset_dir)

    if resolved_dataset_dir is None:
        raise ValueError(
            "Provide either --reference-csv and --current-csv, or use --dataset-dir, "
            "or enable --auto-kagglehub."
        )

    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_dataset_dir}")

    ref_path = find_csv_file(resolved_dataset_dir, "fraudTrain.csv")
    cur_path = find_csv_file(resolved_dataset_dir, "fraudTest.csv")

    data_dir_path = resolve_project_path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    staged_ref_path = data_dir_path / "fraudTrain.csv"
    staged_cur_path = data_dir_path / "fraudTest.csv"

    if ref_path.resolve() != staged_ref_path.resolve():
        shutil.copy2(ref_path, staged_ref_path)
        print(f"Copied reference file to: {staged_ref_path}")
    else:
        print(f"Reference file already in data dir: {staged_ref_path}")

    if cur_path.resolve() != staged_cur_path.resolve():
        shutil.copy2(cur_path, staged_cur_path)
        print(f"Copied current file to: {staged_cur_path}")
    else:
        print(f"Current file already in data dir: {staged_cur_path}")

    return str(staged_ref_path), str(staged_cur_path)


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
    parser.add_argument("--reference-csv", required=False, help="Path to reference CSV")
    parser.add_argument("--current-csv", required=False, help="Path to current batch CSV")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing fraudTrain.csv and fraudTest.csv (or any CSVs as fallback)",
    )
    parser.add_argument(
        "--auto-kagglehub",
        action="store_true",
        help="Download dataset via kagglehub and resolve train/test CSV paths automatically",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="kartik2112/fraud-detection",
        help="Kaggle dataset identifier used with --auto-kagglehub",
    )
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Directory where fraudTrain.csv and fraudTest.csv are stored/staged",
    )
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

    try:
        reference_csv, current_csv = resolve_input_paths(
            reference_csv=args.reference_csv,
            current_csv=args.current_csv,
            dataset_dir=args.dataset_dir,
            auto_kagglehub=args.auto_kagglehub,
            kaggle_dataset=args.kaggle_dataset,
            data_dir=args.data_dir,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        return 1

    if not os.path.exists(reference_csv):
        print(
            f"Reference file not found: {reference_csv}\n"
            f"Current working directory: {os.getcwd()}",
            file=sys.stderr,
        )
        return 1

    if not os.path.exists(current_csv):
        print(
            f"Current file not found: {current_csv}\n"
            f"Current working directory: {os.getcwd()}",
            file=sys.stderr,
        )
        return 1

    if args.bins < 2:
        print("--bins must be >= 2", file=sys.stderr)
        return 1

    if args.psi_warn > args.psi_alert:
        print("--psi-warn cannot be greater than --psi-alert", file=sys.stderr)
        return 1

    features = parse_feature_list(args.features)

    output_csv = str(resolve_project_path(args.output_csv))
    report_df = run_monitoring(
        reference_csv=reference_csv,
        current_csv=current_csv,
        output_csv=output_csv,
        features=features,
        bins=args.bins,
        warn_threshold=args.psi_warn,
        alert_threshold=args.psi_alert,
    )

    if report_df.empty:
        print("No matching features found. Report generated but empty.")
        return 0

    status_counts = report_df["status"].value_counts().to_dict()
    print(f"PSI report written to: {output_csv}")
    print(f"Status summary: {status_counts}")
    print("Top 5 features by PSI:")
    print(report_df[["feature", "psi", "status"]].head(5).to_string(index=False))

    if args.fail_on_alert and (report_df["status"] == "alert").any():
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
