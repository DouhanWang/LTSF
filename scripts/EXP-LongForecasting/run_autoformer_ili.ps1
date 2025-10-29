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
<<<<<<< HEAD
$pred_lens = @(24, 36, 48, 60)
=======
$seq_len = 4
$label_len = 2
$pred_lens = @(4)#@(24, 36, 48, 60)
>>>>>>> 80f2408 (Apply local stashed changes and restore file)

foreach ($pred_len in $pred_lens) {
    Write-Host "Running $model_name on ILI dataset with pred_len = $pred_len ..." -ForegroundColor Cyan

    # python -u run_longExp.py `
    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path national_illness.csv `
<<<<<<< HEAD
        --model_id ili_36_$pred_len `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len 36 `
        --label_len 18 `
=======
        --model_id ili_$seq_len_$pred_len_quantile `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len $seq_len `
        --label_len $label_len `
>>>>>>> 80f2408 (Apply local stashed changes and restore file)
        --pred_len $pred_len `
        --e_layers 2 `
        --d_layers 1 `
        --factor 3 `
        --enc_in 7 `
        --dec_in 7 `
        --c_out 7 `
<<<<<<< HEAD
        --des 'Exp' `
        --itr 1 `
        --use_gpu True `
        --gpu 0 `
        *> "logs/LongForecasting/${model_name}_ili_${pred_len}.log"

    Write-Host "✅ Finished $model_name with pred_len = $pred_len" -ForegroundColor Green
}

Write-Host "🎯 All Autoformer ILI experiments complete!" -ForegroundColor Yellow
=======
        --des 'Exp_Quantile' `
        --loss quantile `
        --quantiles 0.1 0.5 0.9 `
        --itr 1 `
        --use_gpu True `
        --gpu 0 `
        *> "logs/LongForecasting/${model_name}_ili_${seq_len}_${pred_len}_quantile.log"

    Write-Host "Finished $model_name with pred_len = $pred_len" -ForegroundColor Green
}

Write-Host "All Autoformer ILI experiments complete!" -ForegroundColor Yellow
>>>>>>> 80f2408 (Apply local stashed changes and restore file)
