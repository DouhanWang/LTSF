# ============================
# TabPFN_ts Rolling Runner
# ============================

# Create log directories if they don't exist
if (!(Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" | Out-Null }
if (!(Test-Path "./logs/TabPFN_ts")) { New-Item -ItemType Directory -Path "./logs/TabPFN_ts" | Out-Null }



# --- Settings (edit these) ---
$script_path = ".\models\TabPFN_ts.py"
$conda_env = "tabpfn-ts"
# dataset
$root_path = ".\dataset"
$data_path = "per_country_csv\IT.csv"          # your csv
$full_data_path = Join-Path $root_path $data_path

# columns
$item_col  = "item_id"
$target_col = "incidenza"                      # <-- change if needed

# optional: run only one item_id (set to $null to run all)
$item_ids = @(0)                               # e.g. @(0) or @($null)

# rolling config
$test_sizes = @(21)
$pred_lens  = @(4)
$freq_days  = 7                                # weekly timestamps padding
$tabpfn_mode = "client"                        # client | local

# output
$out_root = ".\results\TabPFN_ts"

# ============================
# Run loops
# ============================
foreach ($item_id in $item_ids) {
    foreach ($test_size in $test_sizes) {
        foreach ($pred_len in $pred_lens) {

            $item_tag = if ($null -eq $item_id) { "allItems" } else { "item$item_id" }
            $run_tag = "tabpfn_ts_${item_tag}_test${test_size}_pred${pred_len}"

            $out_dir = Join-Path $out_root $run_tag
            if (!(Test-Path $out_dir)) { New-Item -ItemType Directory -Path $out_dir | Out-Null }

            $log_file = ".\logs\TabPFN_ts\${run_tag}.log"

            Write-Host "Starting TabPFN_ts rolling: $run_tag" -ForegroundColor Cyan
            Write-Host "  data: $full_data_path" -ForegroundColor DarkCyan
            Write-Host "  out : $out_dir" -ForegroundColor DarkCyan
            Write-Host "  log : $log_file" -ForegroundColor DarkCyan

            # Build args
            $args = @(
                "-u", $script_path,
                "--data_path", $full_data_path,
                "--out_dir", $out_dir,
                "--item_col", $item_col,
                "--target", $target_col,
                "--test_size", $test_size,
                "--pred_len", $pred_len,
                "--freq_days", $freq_days,
                "--tabpfn_mode", $tabpfn_mode
            )

            # Add item_id if specified
            if ($null -ne $item_id) {
                $args += @("--item_id", $item_id)
            }

            # Run (redirect both stdout+stderr)
            conda run -n $conda_env --no-capture-output python @args *> $log_file

            Write-Host "Finished: $run_tag" -ForegroundColor Green
        }
    }
}

Write-Host "All TabPFN_ts rolling experiments complete!" -ForegroundColor Yellow
