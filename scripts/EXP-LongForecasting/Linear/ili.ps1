# ===============================================
# PowerShell version of ili.sh
# Run this in PowerShell or PyCharm Terminal
# Make sure to first activate your conda environment:
#     conda activate LTSF_Linear
# ===============================================

# Create log directories if they don’t exist
if (-not (Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (-not (Test-Path "./logs/LongForecasting")) {
    New-Item -ItemType Directory -Path "./logs/LongForecasting" | Out-Null
}

# Define variables
$seq_len = 36 #104
$model_name = "DLinear"

Write-Host "Running experiments for $model_name with seq_len=$seq_len"
Write-Host "============================================================"

# Define prediction lengths
$pred_lengths = @(24, 36, 48, 60)

foreach ($pred in $pred_lengths) {
    $log_path = "logs/LongForecasting/${model_name}_ili_${seq_len}_${pred}.log"
    Write-Host "Starting pred_len=$pred ..."

    python -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path national_illness.csv `
        --model_id "national_illness_${seq_len}_${pred}" `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len $seq_len `
        --label_len 18 `
        --pred_len $pred `
        --enc_in 7 `
        --des "Exp" `
        --itr 1 --batch_size 32 --learning_rate 0.01 `
        *> $log_path

    Write-Host "Finished pred_len=$pred → log saved to $log_path"
    Write-Host "------------------------------------------------------------"
}

Write-Host "✅ All experiments completed!"
