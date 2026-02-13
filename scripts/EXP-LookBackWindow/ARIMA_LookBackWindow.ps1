# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

# --- Your Experiment Settings ---
$model_name = "ARIMA"
$seq_lengths = @(4)
$pred_len = 4
$label_len = 0

foreach ($seq_len in $seq_lengths) {
    Write-Host "Starting Incidenza: Running $model_name on simulated Italy ILI dataset with seq_len=$seq_len ..." -ForegroundColor Cyan

    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path simulated_Italy_ILI_item0.csv `
        --model_id "simulated_Italy_ili_MS_uncertainty_${seq_len}" `
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
        --arima_p 1 `
        --arima_d 1 `
        --arima_q 0 `
        --arima_trend t `
        --arima_maxiter 200 `
        --arima_alpha 0.2 `
        *> "logs/LookBackWindow/${model_name}_simulated_Italy_ili_MS_incidenza_uncertainty_${seq_len}.log"

    Write-Host "Finished $model_name with seq_len=$seq_len" -ForegroundColor Green
}
