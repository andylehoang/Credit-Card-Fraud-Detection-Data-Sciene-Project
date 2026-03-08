# PSI Monitoring

This folder contains a minimal PSI-based drift monitor for batch data.

## Run

### Option A: Ensure `Data/` files exist, then run PSI

Run this from the repository root. It does the following:

- If `Data/fraudTrain.csv` and `Data/fraudTest.csv` already exist: skip download.
- If one or both files are missing: download via `kagglehub` and copy both files into `Data/`.
- Then run `psi_monitor.py` with the local `Data/` files.

`Data/` is created inside the project root (same level as `monitoring/`) and used as persistent local storage.
The script does not process files directly from kagglehub cache.

### PowerShell (Windows)

```powershell
$trainPath = "Data/fraudTrain.csv"
$testPath  = "Data/fraudTest.csv"

if ((-not (Test-Path $trainPath)) -or (-not (Test-Path $testPath))) {
  New-Item -ItemType Directory -Path "Data" -Force | Out-Null

  python -c "import shutil; from pathlib import Path; import kagglehub; p=Path(kagglehub.dataset_download('kartik2112/fraud-detection')); t=next((f for f in p.glob('*.csv') if f.name.lower()=='fraudtrain.csv'), None); s=next((f for f in p.glob('*.csv') if f.name.lower()=='fraudtest.csv'), None); assert t and s, 'fraudTrain.csv or fraudTest.csv not found in downloaded dataset'; shutil.copy2(t, Path('Data')/'fraudTrain.csv'); shutil.copy2(s, Path('Data')/'fraudTest.csv'); print('Copied to Data/:', t, s)"
}

python monitoring/psi_monitor.py `
  --reference-csv $trainPath `
  --current-csv $testPath `
  --features amt,city_pop,unix_time `
  --psi-warn 0.10 `
  --psi-alert 0.25 `
  --output-csv monitoring/psi_report.csv
```

If `kagglehub` is missing:

```powershell
pip install kagglehub
```

### Option B: Use external dataset folder (for example a kagglehub path)

```powershell
python monitoring/psi_monitor.py `
  --dataset-dir "C:\path\to\kagglehub\datasets\kartik2112\fraud-detection\versions\1" `
  --features amt,city_pop,unix_time `
  --psi-warn 0.10 `
  --psi-alert 0.25 `
  --output-csv monitoring/psi_report.csv
```

The script searches this folder for `fraudTrain.csv` and `fraudTest.csv`.
Then it copies them to `Data/fraudTrain.csv` and `Data/fraudTest.csv` and uses these local files.

### Option C: Auto-download via kagglehub

```powershell
python monitoring/psi_monitor.py `
  --auto-kagglehub `
  --kaggle-dataset kartik2112/fraud-detection `
  --data-dir Data `
  --features amt,city_pop,unix_time `
  --output-csv monitoring/psi_report.csv
```

`--data-dir` defaults to `Data`.

If `kagglehub` is missing, install it first:

```powershell
pip install kagglehub
```

### Bash / zsh

```bash
python monitoring/psi_monitor.py \
  --reference-csv Data/fraudTrain.csv \
  --current-csv Data/fraudTest.csv \
  --features amt,city_pop,unix_time \
  --psi-warn 0.10 \
  --psi-alert 0.25 \
  --output-csv monitoring/psi_report.csv
```

Note: `hour`, `day_of_week`, `month`, `age`, `distance_km` are engineered features and do not exist in raw CSV files unless you created them first.

If `--features` is omitted, the script uses all common numeric columns (except `is_fraud`).

## Alerting mode

Use `--fail-on-alert` to return exit code `2` when any feature is above alert threshold.
This is useful in CI or scheduled jobs.

```powershell
python monitoring/psi_monitor.py `
  --reference-csv Data/fraudTrain.csv `
  --current-csv Data/fraudTest.csv `
  --fail-on-alert
```

## Output columns

- `feature`
- `psi`
- `status` (`ok`, `warn`, `alert`, `no_data`)
- missing rates, means, and row counts for reference/current data
