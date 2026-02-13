# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

# --- Your Experiment Settings ---
$model_name = "Naive"
$seq_lengths = @(4)
$pred_len = 4
$label_len = 0
# ---
# for China data
#--features MS
#--target incidence
#--enc_in 1 --dec_in 1 --c_out 1
# Loop for each sequence length
foreach ($seq_len in $seq_lengths) {
    Write-Host "Starting Incidenza: Running $model_name on simulated Italy ILI dataset with seq_len=$seq_len ..." -ForegroundColor Cyan
    # Run the Python experiment
    # PowerShell uses a backtick ` for line continuation
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
           --e_layers 2 `
           --dropout 0.1 `
           --des "Exp" `
           --loss mse `
           --itr 1 `
           --batch_size 32 `
           --learning_rate 0.005 `
           --train_epochs 1 `
           --patience 1 `
           --num_workers 0 `
           --use_gpu True `
           --gpu 0 `
           *> "logs/LookBackWindow/${model_name}_simulated_Italy_ili_MS_incidenza_uncertainty_${seq_len}.log"

    Write-Host "Finished $model_name with seq_len=$seq_len" -ForegroundColor Green
}

Write-Host "All DLinear ILI experiments complete!" -ForegroundColor Yellow