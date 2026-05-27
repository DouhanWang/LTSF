# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

param(
    [int[]]$SeqLengths = @(6, 8),
    [string[]]$Models = @("DLinear", "LSTM", "Autoformer"),
    [string[]]$TrainSettings = @("real", "augmented", "combined"),
    [string[]]$Countries = @("Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"),
    [int]$MaxRetries = 1
)

$ErrorActionPreference = "Continue"

# ---------- paths ----------
$PKG_NAME     = "epi4cast"
$CURRENT_DIR  = (Get-Location).Path

if (Test-Path (Join-Path $CURRENT_DIR "run_longExp.py")) {
    # Running from qm\epi4cast
    $EPICAST_DIR = $CURRENT_DIR
    $PROJECT_ROOT = Split-Path $EPICAST_DIR -Parent
} else {
    # Running from qm
    $PROJECT_ROOT = $CURRENT_DIR
    $EPICAST_DIR = Join-Path $PROJECT_ROOT $PKG_NAME
}

$DATA_ROOT         = Join-Path $EPICAST_DIR "dataset"
$LOG_ROOT          = Join-Path $EPICAST_DIR "logs"
$RESULT_ROOT       = Join-Path $EPICAST_DIR "results"
$TEST_RESULT_ROOT  = Join-Path $EPICAST_DIR "test_results"
$ENV_GPU           = "ltsf-gpu"

# ---------- forecasting config ----------
$PRED_LEN = 4
$DLINEAR_MOVING_AVG = 3
$LABEL_LEN_DEFAULT = 0
$LABEL_LEN_AUTOFORMER = 4

# ---------- common dataset columns ----------
$TARGET_COL = "incidenza"
$FEATURES   = "S"
$FREQ       = "w"
$ENC_IN     = 1
$DEC_IN     = 1
$C_OUT      = 1

function Ensure-Dir($p) {
    if (!(Test-Path $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
    }
}

function Get-DataFile($setting, $countryName) {
    if ($setting -eq "real") {
        return "real_${countryName}_ILI.csv"
    }
    if ($setting -eq "augmented") {
        return "augmented_${countryName}_ILI.csv"
    }
    if ($setting -eq "combined") {
        return "combined_${countryName}_ILI.csv"
    }
    throw "Unknown train setting: $setting"
}

function Invoke-ExperimentPython($pythonArgs, $logFile) {
    $env:PYTHONPATH = $PROJECT_ROOT

    if ($env:CONDA_DEFAULT_ENV -eq $ENV_GPU) {
        Push-Location $EPICAST_DIR
        try {
            & python -u @pythonArgs *> $logFile
            return $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }

    conda run -n $ENV_GPU --cwd $EPICAST_DIR --no-capture-output python -u @pythonArgs *> $logFile
    return $LASTEXITCODE
}

function Run-LongExpModel($modelName, $dataFile, $countryName, $seqLen) {
    $tag = ($dataFile -replace "\.csv$", "")
    $labelLen = if ($modelName -eq "Autoformer") { $LABEL_LEN_AUTOFORMER } else { $LABEL_LEN_DEFAULT }
    $modelId = "{0}_{1}_{2}" -f $modelName, $tag, $seqLen

    $args = @(
        "-m", "$PKG_NAME.run_longExp",
        "--is_training", "1",
        "--root_path", $DATA_ROOT,
        "--data_path", $dataFile,
        "--model_id", $modelId,
        "--model", $modelName,
        "--data", "custom",
        "--features", $FEATURES,
        "--target", $TARGET_COL,
        "--freq", $FREQ,
        "--seq_len", "$seqLen",
        "--label_len", "$labelLen",
        "--pred_len", "$PRED_LEN",
        "--enc_in", "$ENC_IN",
        "--dec_in", "$DEC_IN",
        "--c_out", "$C_OUT",
        "--des", "Exp",
        "--loss", "mse",
        "--itr", "1",
        "--num_workers", "0",
        "--use_gpu", "True",
        "--gpu", "0"
    )

    if ($modelName -eq "DLinear") {
        $args += @(
            "--moving_avg", "$DLINEAR_MOVING_AVG",
            "--batch_size", "32",
            "--learning_rate", "0.005",
            "--train_epochs", "30",
            "--patience", "3"
        )
    }
    elseif ($modelName -eq "LSTM") {
        $args += @(
            "--e_layers", "2",
            "--dropout", "0.1",
            "--batch_size", "32",
            "--learning_rate", "0.005",
            "--train_epochs", "30",
            "--patience", "3"
        )
    }
    elseif ($modelName -eq "Autoformer") {
        $args += @(
            "--e_layers", "2",
            "--d_layers", "1",
            "--factor", "3",
            "--moving_avg", "5",
            "--d_model", "32",
            "--d_ff", "64",
            "--n_heads", "2",
            "--dropout", "0.05",
            "--batch_size", "32",
            "--learning_rate", "0.0005",
            "--train_epochs", "30",
            "--patience", "3"
        )
    }
    else {
        Write-Host "SKIP (unsupported model in this script): $modelName" -ForegroundColor Yellow
        return
    }

    $logDir = Join-Path $LOG_ROOT "LookBackWindow"
    Ensure-Dir $logDir
    $logFile = Join-Path $logDir ("{0}_{1}_{2}.log" -f $modelName, $tag, $seqLen)

    Write-Host "Running $modelName | $countryName | $dataFile | lookback=$seqLen" -ForegroundColor Cyan
    Write-Host "  result folder: results/$modelId" -ForegroundColor DarkCyan
    Write-Host "  log: $logFile" -ForegroundColor DarkCyan

    for ($attempt = 0; $attempt -le $MaxRetries; $attempt++) {
        if ($attempt -gt 0) {
            Write-Host "Retrying $modelName | $countryName | $dataFile | lookback=$seqLen (attempt $($attempt + 1) of $($MaxRetries + 1))" -ForegroundColor Yellow
        }

        $exitCode = Invoke-ExperimentPython $args $logFile
        if ($exitCode -eq 0) {
            Write-Host "Finished $modelName | $countryName | $dataFile | lookback=$seqLen" -ForegroundColor Green
            return
        }

        Write-Host "FAILED $modelName | $countryName | $dataFile | lookback=$seqLen (see $logFile)" -ForegroundColor Red
    }
}

Ensure-Dir $LOG_ROOT
Ensure-Dir $RESULT_ROOT
Ensure-Dir $TEST_RESULT_ROOT

foreach ($seqLen in $SeqLengths) {
    foreach ($country in $Countries) {
        foreach ($model in $Models) {
            foreach ($setting in $TrainSettings) {
                $dataFile = Get-DataFile $setting $country
                $filePath = Join-Path $DATA_ROOT $dataFile

                if (!(Test-Path $filePath)) {
                    Write-Host "SKIP (missing): $filePath" -ForegroundColor Yellow
                    continue
                }

                Run-LongExpModel $model $dataFile $country $seqLen
            }
        }
    }
}

Write-Host "All deep-model look-back-window experiments complete!" -ForegroundColor Green
