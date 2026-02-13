# Create log directories if they don't exist
if (!(Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (!(Test-Path "./logs/LookBackWindow")) {
    New-Item -ItemType Directory -Path "./logs/LookBackWindow" | Out-Null
}

$model_name = "Autoformer"
$seq_lengths = @(4) # , 6, 8

$pred_len = 4
$label_len = 4
$moving_avg=@(5) #3
# for China data
#--features MS
#--target incidence
#--enc_in 1 --dec_in 1 --c_out 1
# Inner loop for sequence length
foreach ($seq_len in $seq_lengths) {
       foreach($moving_avg in $moving_avg)
       {
              Write-Host "Starting INCIDENZA: Running $model_name on combined Italy ILI dataset with seq_len=$seq_len, moving_avg = $moving_avg ..." -ForegroundColor Cyan
              # Run the Python experiment
              # PowerShell uses a backtick ` for line continuation
              # model_id A原本是MS
              & "C:\Users\Douhan\anaconda3\envs\ltsf-gpu\python.exe" -u run_longExp.py `
           --is_training 1 `
           --root_path ./dataset/ `
           --data_path combined_Italy_ILI.csv `
           --model_id "combined_Italy_ili_S_incidenza_sdscaler_uncertainty_${seq_len}_${moving_avg}" `
           --model "$model_name" `
           --data custom `
           --features S `
           --target incidenza `
           --seq_len $seq_len `
           --label_len $label_len `
           --pred_len $pred_len `
           --e_layers 2 `
           --d_layers 1 `
           --factor 3 `
           --moving_avg $moving_avg `
           --enc_in 1 `
           --dec_in 1 `
           --c_out 1 `
           --d_model 32 `
           --d_ff 64 `
           --n_heads 2 `
           --dropout 0.05 `
           --des "Exp" `
           --loss mse `
           --itr 1 `
           --batch_size 32 `
           --learning_rate 0.0005 `
           --train_epochs 30 `
           --num_workers 0 `
           --patience 3 `
           --freq w `
           --use_gpu True `
           --gpu 0 `
           *> "logs/LookBackWindow/${model_name}_combined_Italy_ili_S_incidenza_sdscaler_uncertainty_${seq_len}_${moving_avg}.log"
              Write-Host "Finished $model_name with seq_len = $seq_len, moving_avg = $moving_avg" -ForegroundColor Green
       }
}

Write-Host "All Autoformer ILI experiments complete!" -ForegroundColor Yello w

#           --features MS `
#           --seq_len $seq_len `
#           --label_len $label_len `
#           --pred_len $pred_len `
#           --e_layers 2 `
#           --d_layers 1 `
#           --factor 3 `
#           --enc_in 1 `
#           --dec_in 1 `
#           --c_out 1 `
#           --d_model 64 `
#           --d_ff 128 `
#           --n_heads 4 `
#           --des 'Exp' `
#           --loss mse `
#           --itr 1 `
#           --batch_size 4 `
#           --learning_rate 0.001 `
#           --use_gpu True `
#           --gpu 0 `