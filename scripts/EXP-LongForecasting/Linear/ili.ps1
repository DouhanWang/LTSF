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

if (-not (Test-Path "./logs/ShortForecasting")) {
    New-Item -ItemType Directory -Path "./logs/ShortForecasting" | Out-Null
}

# Define variables
$seq_len = 4 #104
$model_name = "DLinear"

Write-Host "Running experiments for $model_name with seq_len=$seq_len"
Write-Host "============================================================"

# Define prediction lengths
$pred_lengths = @(4)

foreach ($pred in $pred_lengths) {
    $log_path = "logs/ShortForecasting/${model_name}_italy_ili_${seq_len}_${pred}.log"
    Write-Host "Starting pred_len=$pred ..."

    python -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path italia_2025_17_ILI.csv `
        --model_id "Dlinear_italy_ili_${seq_len}_${pred}_quantile" `
        --model $model_name `
        --data custom `
        --features M `
        --seq_len $seq_len `
        --label_len 2 `
        --pred_len $pred `
        --enc_in 9 `
        --des "Exp_Quantile" `
        --itr 1 --batch_size 1 --learning_rate 0.01 `
        *> $log_path

    Write-Host "Finished pred_len=$pred → log saved to $log_path"
    Write-Host "------------------------------------------------------------"
}

Write-Host "All experiments completed!"
