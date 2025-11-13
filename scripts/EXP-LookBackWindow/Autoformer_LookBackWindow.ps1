# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

$model_name = "Autoformer"
$seq_lengths = @(4, 6, 8)

$pred_len = 4



# Inner loop for sequence length
foreach ($seq_len in $seq_lengths) {
    Write-Host "Starting INCIDENZA: Running $model_name on Italy ILI dataset with seq_len=$seq_len, pred_len = $pred_len ..." -ForegroundColor Cyan
    $label_len = [Math]::Floor($seq_len / 2)
    # Run the Python experiment
    # PowerShell uses a backtick ` for line continuation
    & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
           --is_training 1 `
           --root_path ./dataset/ `
           --data_path italia_17_25_ILI.csv `
           --model_id italy_ili_MS_incidenza_uncertainty_$seq_len_$pred_len `
           --model "$model_name" `
           --data custom `
           --features MS `
           --seq_len $seq_len `
           --label_len $label_len `
           --pred_len $pred_len `
           --e_layers 2 `
           --d_layers 1 `
           --factor 3 `
           --enc_in 2 `
           --dec_in 2 `
           --c_out 2 `
           --d_model 64 `
           --d_ff 128 `
           --n_heads 4 `
           --des 'Exp' `
           --loss mse `
           --itr 1 `
           --batch_size 4 `
           --use_gpu True `
           --gpu 0 `
           *> "logs/LookBackWindow/${model_name}_italy_ili_MS_incidenza_uncertainty_${seq_len}_${pred_len}.log"
    Write-Host "Finished $model_name with seq_len = $seq_len, pred_len = $pred_len" -ForegroundColor Green
}

Write-Host "All Autoformer Italy ILI experiments complete!" -ForegroundColor Yellow