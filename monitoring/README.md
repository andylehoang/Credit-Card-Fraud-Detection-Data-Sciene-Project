# PSI Monitoring

This folder contains a minimal PSI-based drift monitor for batch data.

## Run

```bash
python monitoring/psi_monitor.py \
  --reference-csv Data/fraudTrain.csv \
  --current-csv Data/fraudTest.csv \
  --features amt,city_pop,hour,day_of_week,month,age,distance_km \
  --psi-warn 0.10 \
  --psi-alert 0.25 \
  --output-csv monitoring/psi_report.csv
```

If `--features` is omitted, the script uses all common numeric columns (except `is_fraud`).

## Alerting mode

Use `--fail-on-alert` to return exit code `2` when any feature is above alert threshold.
This is useful in CI or scheduled jobs.

```bash
python monitoring/psi_monitor.py \
  --reference-csv Data/fraudTrain.csv \
  --current-csv Data/fraudTest.csv \
  --fail-on-alert
```

## Output columns

- `feature`
- `psi`
- `status` (`ok`, `warn`, `alert`, `no_data`)
- missing rates, means, and row counts for reference/current data
