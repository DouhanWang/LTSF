# ============================
# Run_AllCountries_1to4step.ps1
# ============================
$ErrorActionPreference = "Continue"
# ---------- paths ----------
$PROJECT_ROOT = (Get-Location).Path              # should be qm/
$EPICAST_DIR = Join-Path $PROJECT_ROOT $PKG_NAME   # qm\epi4cast
$PKG_NAME     = "epi4cast"
$DATA_ROOT    = Join-Path $PROJECT_ROOT "$PKG_NAME\dataset"   # <-- adjust if your datasets live elsewhere
$LOG_ROOT     = Join-Path $PROJECT_ROOT "$PKG_NAME\logs"
$RESULT_ROOT  = Join-Path $PROJECT_ROOT "$PKG_NAME\results"
$TEST_RESULT_ROOT  = Join-Path $PROJECT_ROOT "$PKG_NAME\test_results"
$ENV_GPU   = "ltsf-gpu"
# ---------- conda env for TabPFN_ts ----------
$ENV_TABPFN = "tabpfn-ts"     # change if your env name differs
$TABPFN_MODE  = "client"        # client | local
$FREQ_DAYS    = 7

# ---------- forecasting config ----------
$SEQ_LEN   = 4
$PRED_LEN  = 4
$LABEL_LEN_DEFAULT = 0          # for most models
$LABEL_LEN_AUTOFORMER = 4       # Autoformer script used label_len=4

# ---------- common dataset columns ----------
$TARGET_COL = "incidenza"
$FEATURES   = "S"
$FREQ       = "w"
$ENC_IN     = 1
$DEC_IN     = 1
$C_OUT      = 1

# ---------- countries (file-name Country part) ----------
# These should match your filenames exactly: real_Belgium_ILI.csv, etc.
$countries = @("Poland") #"Belgium","Czechia","Denmark","France","Ireland","Italy","Netherlands","Poland","Romania"

# ---------- datasets ----------
function Get-DatasetsForModel($modelName, $countryName) {
    $real     = "real_${countryName}_ILI.csv"
    $sim      = "simulated_${countryName}_ILI.csv"
    $aug      = "augmented_${countryName}_ILI.csv"
    $median   = "simulated_${countryName}_ILI_median.csv"
    $combined = "combined_${countryName}_ILI.csv"

    if ($modelName -in @("ARIMA","TabPFN_ts")) {
        return @($real, $median)   # only real + median
    } else {
        return @($real, $sim, $aug, $combined)  # no median
    }
}

# ---------- models you want to run ----------
# Add more here if needed (but keep the "median rule" in mind).
$models = @("TabPFN_ts") #"DLinear","Autoformer","LSTM","Naive","ARIMA",

# ---------- helpers ----------
function Ensure-Dir($p) {
    if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

Ensure-Dir $LOG_ROOT
Ensure-Dir $RESULT_ROOT
Ensure-Dir $TEST_RESULT_ROOT

function Run-LongExpModel($modelName, $dataFile, $countryName) {
    $dataPath = $dataFile
        $tag = ($dataFile -replace "\.csv$","")
    # label_len differs for Autoformer
    $labelLen = if ($modelName -eq "Autoformer") { $LABEL_LEN_AUTOFORMER } else { $LABEL_LEN_DEFAULT }

    # run_longExp.py args (based on your existing LookBackWindow scripts)
    $args = @(
        "-m", "$PKG_NAME.run_longExp",
        "--is_training", "1",
        "--root_path", $DATA_ROOT,
        "--data_path", $dataPath,
        "--model_id", ("{0}_{1}" -f $modelName, $tag),
        "--model", $modelName,
        "--data", "custom",
        "--features", $FEATURES,
        "--target", $TARGET_COL,
        "--freq", $FREQ,
        "--seq_len", $SEQ_LEN,
        "--label_len", $labelLen,
        "--pred_len", $PRED_LEN,
        "--enc_in", $ENC_IN,
        "--des", "Exp",
        "--loss", "mse",
        "--itr", "1",
        "--num_workers", "0",
        "--use_gpu", "True",
        "--gpu", "0"
    )

    # Per-model hyperparams (copied from your scripts)
    if ($modelName -eq "DLinear") {
        $args += @("--batch_size","32","--learning_rate","0.005","--train_epochs","30","--patience","3")
    }
    elseif ($modelName -eq "LSTM") {
        $args += @("--e_layers","2","--dropout","0.1","--batch_size","32","--learning_rate","0.005","--train_epochs","30","--patience","3")
    }
    elseif ($modelName -eq "Naive") {
        $args += @("--e_layers","2","--dropout","0.1","--batch_size","32","--learning_rate","0.005","--train_epochs","1","--patience","1")
    }
    elseif ($modelName -eq "Autoformer") {
        # use the same defaults as your Autoformer_LookBackWindow.ps1
        $moving_avg = 5
        $args += @(
            "--e_layers","2",
            "--d_layers","1",
            "--factor","3",
            "--moving_avg","$moving_avg",
            "--dec_in",$DEC_IN,
            "--c_out",$C_OUT,
            "--d_model","32",
            "--d_ff","64",
            "--n_heads","2",
            "--dropout","0.05",
            "--batch_size","32",
            "--learning_rate","0.0005",
            "--train_epochs","30",
            "--patience","3"
        )
        # make model_id include moving_avg for clarity
        $args = $args | ForEach-Object { $_ }
    }
    elseif ($modelName -eq "ARIMA") {
        # keep your ARIMA settings
        $args += @(
            "--batch_size","1",
            "--learning_rate","0.005",
            "--train_epochs","1",
            "--patience","1",
            "--arima_p","1",
            "--arima_d","1",
            "--arima_q","0",
            "--arima_alpha","0.2"
        )
    }

    $logDir = Join-Path $LOG_ROOT $modelName
    Ensure-Dir $logDir


    $logFile = Join-Path $logDir ("{0}_{1}.log" -f $countryName,$tag)

    Write-Host "Running $modelName | $countryName | $dataFile" -ForegroundColor Cyan
    Write-Host "  log: $logFile" -ForegroundColor DarkCyan

    # run (stdout+stderr to log)
    # --- run_longExp should save under epi4cast/ (results/test_results/checkpoints) ---
#     Push-Location $EPICAST_DIR
    $env:PYTHONPATH = $PROJECT_ROOT   # ensure "import epi4cast" works even when cwd=epi4cast

    conda run -n $ENV_GPU --cwd $EPICAST_DIR --no-capture-output python -u @args *> $logFile

#     Pop-Location
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED $modelName | $countryName | $dataFile (see $logFile)" -ForegroundColor Red }
}

function Run-TabPFN($dataFile, $countryName) {
    # TabPFN_ts expects full data_path
    $fullDataPath = Join-Path $DATA_ROOT $dataFile
    # 2) setting name you want: real_Italy_ILI_TabPFN_ts
    $tag = ($dataFile -replace "\.csv$","")         # e.g. real_Italy_ILI
    $setting = "TabPFN_ts_$tag"                   # e.g. real_Italy_ILI_TabPFN_ts

    # 3) output dirs inside epi4cast/results and epi4cast/test_results
    $outDir  = Join-Path $RESULT_ROOT $setting
    $testDir = Join-Path $TEST_RESULT_ROOT $setting
    Ensure-Dir $outDir
    Ensure-Dir $testDir


    $logDir = Join-Path $LOG_ROOT "TabPFN_ts"
    Ensure-Dir $logDir
    $logFile = Join-Path $logDir ("{0}_{1}.log" -f $countryName, ($dataFile -replace "\.csv$",""))
    # 5) run in epi4cast cwd so relative paths behave, but keep import working
    Push-Location $EPICAST_DIR
    $env:PYTHONPATH = $PROJECT_ROOT
    # NOTE: using module run for your packaged project
    $args = @(
        "-m", "$PKG_NAME.models.TabPFN_ts",
        "--data_path", $fullDataPath,
        "--out_dir", $outDir,
        "--item_col", "item_id",
        "--target", $TARGET_COL,
        "--test_size", "21",
        "--pred_len", "$PRED_LEN",
        "--freq_days", "$FREQ_DAYS",
        "--tabpfn_mode", $TABPFN_MODE,
        "--item_id", "0"
    )

    Write-Host "Running TabPFN_ts | $countryName | $dataFile" -ForegroundColor Cyan
    Write-Host "  out: $outDir" -ForegroundColor DarkCyan
    Write-Host "  log: $logFile" -ForegroundColor DarkCyan
    conda run -n $ENV_TABPFN --no-capture-output python @args *> $logFile
    Pop-Location
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED TabPFN_ts | $countryName | $dataFile (see $logFile)" -ForegroundColor Red }
}

# ============================
# Main loops
# ============================
foreach ($country in $countries) {
    foreach ($model in $models) {

        $datasets = Get-DatasetsForModel $model $country

        foreach ($ds in $datasets) {
            $filePath = Join-Path $DATA_ROOT $ds
            if (!(Test-Path $filePath)) {
                Write-Host "SKIP (missing): $filePath" -ForegroundColor Yellow
                continue
            }

            if ($model -eq "TabPFN_ts") {
                Run-TabPFN $ds $country
            } else {
                Run-LongExpModel $model $ds $country
            }
        }
    }
}

Write-Host "All experiments complete!" -ForegroundColor Green