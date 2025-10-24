# Run Autoformer on the ILI (national_illness.csv) dataset
# This script creates logs and runs all desired prediction lengths

# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LongForecasting")) {
    New-Item -ItemType Directory -Path "./logs/LongForecasting" | Out-Null
}

# Only one model and a few prediction lengths
$model_name = "Autoformer"
$pred_lens = @(24, 36, 48, 60)

foreach ($pred_len in $pred_lens) {
    Write-Host "Running $model_name on ILI dataset with pred_len = $pred_len ..." -ForegroundColor Cyan

    # python -u run_longExp.py `
    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path national_illness.csv `
        --model_id ili_36_$pred_len `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len 36 `
        --label_len 18 `
        --pred_len $pred_len `
        --e_layers 2 `
        --d_layers 1 `
        --factor 3 `
        --enc_in 7 `
        --dec_in 7 `
        --c_out 7 `
        --des 'Exp' `
        --itr 1 `
        --use_gpu True `
        --gpu 0 `
        *> "logs/LongForecasting/${model_name}_ili_${pred_len}.log"

    Write-Host "✅ Finished $model_name with pred_len = $pred_len" -ForegroundColor Green
}

Write-Host "🎯 All Autoformer ILI experiments complete!" -ForegroundColor Yellow
