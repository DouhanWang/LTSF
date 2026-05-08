# From Naive to Foundation: Benchmarking Models for Epidemic Forecasting

This repository contains the PyTorch implementation of epi4cast for the manuscript "From Naive to Foundation: Benchmarking Models for Epidemic Forecasting".

The code benchmarks statistical, neural, and foundation-model approaches for weekly influenza-like illness (ILI) forecasting across European countries. The main forecasting setup uses rolling 1- to 4-week-ahead evaluation over influenza-season windows.

## Updates

- 2026-05-08: Updated the ARIMA pipeline to use non-seasonal auto-ARIMA with rolling-origin refitting.
- 2026-05-08: Cleaned the experiment scripts and standardized look-back-window experiments across countries.

## Implemented Models

The main experiments include:

- Naive baseline
- Auto-ARIMA
- DLinear
- LSTM
- Autoformer
- TabPFN-TS
- Weighted and unweighted ensembles
- RespiCast hub ensemble baseline

## Repository Structure

| Path | Description |
| --- | --- |
| `data_provider/` | Data loading and split logic for ILI forecasting datasets. |
| `models/` | Forecasting model implementations. |
| `exp/` | Training, rolling evaluation, metric computation, and prediction export logic. |
| `scripts/EXP-LookBackWindow/` | PowerShell scripts for look-back-window experiments across countries and models. |
| `dataset/` | Input data directory. Raw data and processed country-level CSV files should be placed here. |
| `results/` | Saved predictions, metrics, and generated metric tables. |
| `test_results/` | Forecast plots and diagnostic figures. |


## Environment

First, install Conda. The main PyTorch environment can be created with:

```bash
conda create -n ltsf-gpu python=3.10 -y
conda activate ltsf-gpu
pip install -r requirements-core.txt
pip install -r requirements-torch-cu121.txt
```

TabPFN-TS uses a separate environment:

```bash
conda create -n tabpfn-ts python=3.10 -y
conda activate tabpfn-ts
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-tabpfn.txt
```

To verify the TabPFN-TS installation:

```bash
python -c "import tabpfn_time_series; print('OK')"
```

## Data Preparation

Place all raw and processed data files under `./dataset`.

Primary data sources:

- RespiCast hub ensemble forecasts for ILI/ARI 2024-2025: https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/tree/main/model-output/respicast-hubEnsemble
- ILI incidence for 2017-2018 and 2018-2019: https://github.com/european-modelling-hubs/flu-forecast-hub_archive/blob/main/target-data/latest-ILI_incidence.csv
- ILI incidence for 2023-2024 and 2024-2025: https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/blob/main/target-data/latest-ILI_incidence.csv

ILI incidence is measured per 100,000 population.

Useful preprocessing scripts:

- `extract_raw_data.py`: extract target ILI data from raw surveillance files.
- `extract_respicast_point_pred.py` and `extract_respicast_wis_point_to_npy.py`: extract RespiCast point forecasts and interval-score inputs.
- `check_missing_wis_rows.py`: check missing RespiCast WIS rows.
- `recompute_wis.py`: recompute IS80/WIS80 where needed.
- `data_handling.py`: convert simulated exogenous `.npy` data into CSV format.
- `data_combine.py`: combine observed and simulated trajectories into combined datasets.
- `data_augmentation.py`: generate endogenous augmented trajectories and combine them with observed data.
- `plot_dataset_series.py`: visualize the processed country-level time series before training.

## Running Experiments

The main experiment scripts are in `scripts/EXP-LookBackWindow/`.

To run one model script on Windows, for example Autoformer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\EXP-LookBackWindow\Autoformer_LookBackWindow.ps1
```

To run the all-country experiment script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\EXP-LookBackWindow\run_all_countries.ps1
```

The `run_all_countries.ps1` script contains the active model list and dataset settings. Edit the `$models` variable to select the models to run.

TabPFN-TS should be run in the `tabpfn-ts` environment:

```powershell
conda run -n tabpfn-ts --no-capture-output powershell -ExecutionPolicy Bypass -File scripts\EXP-LookBackWindow\run_all_countries.ps1
```

## Post-Processing and Evaluation

After training and prediction:

- `postprocess_nonneg_dlinear.py`: enforce non-negative DLinear predictions.
- `fix_column_names_preds.py`: standardize prediction file column names.
- `fix_wis_length.py`: check and fix interval-score vector lengths.
- `Ensemble.py`: build the weighted ensemble from model predictions.
- `Unweighted_Ensemble.py`: build the unweighted ensemble baseline.
- `recompute_wis.py`: compute IS80/WIS80 for model and ensemble predictions.
- `all_metrics.py`: compute point and interval metrics.
- `make_metrics_tables_latex.py`: generate LaTeX metric tables.
- `draw_forecast.py`: plot forecasts for all countries and horizons.
- `draw_forecast_best_model.py`: compare RespiCast and the best-performing model.
- `WIS_plot.py`: plot relative interval-score results.
- `rAE_plot.py`: plot relative absolute-error results.
- `plot_win_counts.py`: plot model win counts across countries and horizons.

## Citation

If you find this repository useful, please cite the manuscript:

```bibtex
@article{epi4cast2026,
  title = {From Naive to Foundation: Benchmarking Models for Epidemic Forecasting},
  author = {},
  journal = {},
  year = {2026}
}
```

This project is built upon [[DLinear's GitHub Link](https://github.com/cure-lab/LTSF-Linear)] and has been modified and extended with new features under the Apache License 2.0. The copyright of the original code belongs to the DLinear Authors.
