# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

# --- Your Experiment Settings ---
$model_name = "DLinear"
$seq_lengths = @(4, 6, 8)
$pred_len = 4
# ---

# Loop for each sequence length
foreach ($seq_len in $seq_lengths) {
    Write-Host "Starting Incidenza: Running $model_name on Italy ILI dataset with seq_len=$seq_len, pred_len=$pred_len ..." -ForegroundColor Cyan

    # Dynamically set label_len (e.g., 18 is too large for seq_len=4)
    $label_len = [Math]::Floor($seq_len / 2)

    # Run the Python experiment
    # PowerShell uses a backtick ` for line continuation
    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
           --is_training 1 `
           --root_path ./dataset/ `
           --data_path italia_17_25_ILI.csv `
           --model_id "italy_ili_MS_uncertainty_${seq_len}_${pred_len}" `
           --model "$model_name" `
           --data custom `
           --features MS `
           --seq_len $seq_len `
           --label_len $label_len `
           --pred_len $pred_len `
           --enc_in 2 `
           --des 'Exp' `
           --loss mse `
           --itr 1 `
           --batch_size 4 `
           --learning_rate 0.05 `
           *> "logs/LookBackWindow/${model_name}_italy_ili_MS_incidenza_uncertainty_${seq_len}_${pred_len}.log"

    Write-Host "Finished $model_name with seq_len=$seq_len, pred_len=$pred_len" -ForegroundColor Green
}

Write-Host "All DLinear ILI experiments complete!" -ForegroundColor Yellow