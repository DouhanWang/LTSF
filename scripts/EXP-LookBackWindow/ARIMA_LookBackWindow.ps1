# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

# --- Your Experiment Settings ---
$PY = (Get-Command python).Source
$model_name = "ARIMA"
$seq_lengths = @(4)
$pred_len = 4
$label_len = 0
$arima_start_p = 2
$arima_start_q = 2
$arima_max_p = 5
$arima_max_d = 2
$arima_min_q = 0   # normal auto ARIMA can choose no MA term if q=0 is best
$arima_max_q = 5
$arima_ic = "aic"
$arima_test = "kpss"
$arima_max_order = 5

foreach ($seq_len in $seq_lengths) {
    Write-Host "Starting Incidenza: Running standard auto ARIMA on real Italy ILI dataset with seq_len=$seq_len ..." -ForegroundColor Cyan

    & $PY -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path per_country_csv/IT.csv `
        --model_id "real_Italy_ili_MS_uncertainty_${seq_len}" `
        --model "$model_name" `
        --data custom `
        --features S `
        --target incidenza `
        --freq w `
        --seq_len $seq_len `
        --label_len $label_len `
        --pred_len $pred_len `
        --enc_in 1 `
        --des "Exp" `
        --loss mse `
        --itr 1 `
        --batch_size 1 `
        --learning_rate 0.005 `
        --train_epochs 1 `
        --patience 1 `
        --num_workers 0 `
        --use_gpu True `
        --gpu 0 `
        --arima_auto `
        --arima_start_p $arima_start_p `
        --arima_start_q $arima_start_q `
        --arima_min_p 0 `
        --arima_max_p $arima_max_p `
        --arima_min_d 0 `
        --arima_max_d $arima_max_d `
        --arima_min_q $arima_min_q `
        --arima_max_q $arima_max_q `
        --arima_ic $arima_ic `
        --arima_test $arima_test `
        --arima_max_order $arima_max_order `
        --arima_alpha 0.2 `
        *> "logs/LookBackWindow/${model_name}_real_Italy_ili_MS_incidenza_uncertainty_${seq_len}.log"

    Write-Host "Finished $model_name with seq_len=$seq_len" -ForegroundColor Green
}
