<<<<<<< HEAD
# From naive to foundation: benchmarking models for epidemic forecasting

This repo is the official Pytorch implementation of epi4cast: "[From naive to foundation: benchmarking models for epidemic forecasting](https://)". 


## Updates
- [2026/05/08] We update some scripts for ARIMA. 
  - .




## Features
- [x] Support scripts on different [look-back window size](https://github.com/cure-lab/DLinear/tree/main/scripts/EXP-LookBackWindow).
- [x] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)


## Detailed Description
We provide all experiment script files in `./scripts`:
| Files      |                              Interpretation                          |
| ------------- | -------------------------------------------------------| 
| EXP-LookBackWindow      | Study the impact of different look-back window sizes   | 


This code is simply built on the code base of Autoformer. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

The implementation of Autoformer is from https://github.com/thuml/Autoformer

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n ltsf-gpu python=3.10 -y
conda activate ltsf-gpu
pip install -r requirements.txt
```

TabPFN-ts uses another environment, it can be installed by:
```
conda create -n tabpfn-ts python=3.10 -y
conda activate tabpfn-ts
python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir --prefer-binary "numpy==1.26.4" "tqdm==4.67.1" "huggingface-hub>=0.34,<1.0" "datasets>=2.15,<4.0"
python -m pip install --no-cache-dir --prefer-binary tabpfn-time-series==1.0.8
```

And you can validate if the environment "tabpfn-ts" is installed by
```
python -c "import tabpfn_time_series; print('OK')"
```
### Data Preparation

Respicast Ensemble forecasts: 

ILI/ARI 2024/25: https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/tree/main/model-output/respicast-hubEnsemble

ILI datasaet:For seasons 2017-2018，2018-2019, https://github.com/european-modelling-hubs/flu-forecast-hub_archive/blob/main/target-data/latest-ILI_incidence.csv，
             For seasons 2023-2024，2024-2025, https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/blob/main/target-data/latest-ILI_incidence.csv source from ERVISS
Incidence means per 100,000 population

**Please put them in the `./dataset` directory**

### Training Example
- In `scripts/ `, we provide the model implementation *Dlinear/Autoformer/Informer/Transformer*

For example:

In order to run it on Windows, you can first generate a ps1 version script and use the following code: 
```
powershell -ExecutionPolicy Bypass -File scripts\EXP-LookBackWindow\Autoformer_LookBackWindow.ps1
```
To run all the models across all countries, use
```
cd C:\Users\15952\Desktop\qm
powershell -ExecutionPolicy Bypass -File epi4cast\scripts\EXP-LookBackWindow\run_all_countries.ps1
```

If you want to use specific environment, you can
```
conda run -n tabpfn-ts --no-capture-output powershell -ExecutionPolicy Bypass -File .\epi4cast\scripts\EXP-LookBackWindow\run_all_countries.ps1
```
Before training, use plot_dataset_series.py to visualize data.

After training and prediction, use postprocess_nonneg_dlinear.py to make sure all the predictions are non-negative.

Use extract_respicast_point_pred.npy and extract_respicast_wis_point_to_npy.py to get predictions from Respicast, and use check_missing_wis_rows.py to check missing wis in Repicast，we found that there are some missing values for wis, so we use recompute_wis.py to recompute the wis80 for Respicast.

Use fix_column_names_preds.py to fix column names for all prediction files and use fix_wis_length.py to make sure all wis have correct length.

Use Ensemble.py to get pred of ensemble of our models and recompute__wis.py to compute wis80 for the ensemble.

Use all_metrics.py to compute csv for all metrics and make_metrics_tables_latex.py to get a latex table.

Use draw_forecast.py to get point forecast for all countries for all steps, use draw_forecast_best_model.py to get point forecast for Respicast and our best model, use WIS_plot.py and rAE_plot.py to get plots for relative IS and rAE.

Use plot_win_counts.py to plot win counts






## Citing

If you find this repository useful for your work, please consider citing it as follows:

```BibTeX

```

Please remember to cite all the datasets and compared methods if you use them in your experiments.






