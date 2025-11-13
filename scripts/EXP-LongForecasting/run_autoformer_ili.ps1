# Run Autoformer on the ILI (national_illness.csv) dataset
# This script creates logs and runs all desired prediction lengths

# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/ShortForecasting")) {
    New-Item -ItemType Directory -Path "./logs/ShortForecasting" | Out-Null
}

# Only one model and a few prediction lengths
$model_name = "Autoformer"
$seq_len = 4
$label_len = 2
$pred_lens = @(4)#@(24, 36, 48, 60)

foreach ($pred_len in $pred_lens) {
    Write-Host "Running $model_name on Italy ILI dataset with pred_len = $pred_len ..." -ForegroundColor Cyan

    # python -u run_longExp.py `
    # dec_in,c_out 9(3 featuresx 3 quantiles) `
    #15952 or Douhan
    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path italia_17_25_ILI.csv `
        --model_id italy_ili_$seq_len_$pred_len_quantile `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len $seq_len `
        --label_len $label_len `
        --pred_len $pred_len `
        --e_layers 2 `
        --d_layers 1 `
        --factor 3 `
        --enc_in 3 `
        --dec_in 9 `
        --c_out 9 `
        --d_model 64 `
        --d_ff 128 `
        --n_heads 4 `
        --des 'Exp_Quantile' `
        --loss quantile `
        --quantiles 0.1 0.5 0.9 `
        --itr 1 `
        --batch_size 4 `
        --use_gpu True `
        --gpu 0 `
        *> "logs/ShortForecasting/${model_name}_italy_ili_${seq_len}_${pred_len}_quantile.log"

    Write-Host "Finished $model_name with pred_len = $pred_len" -ForegroundColor Green
}

Write-Host "All Autoformer Italy ILI experiments complete!" -ForegroundColor Yellow

