import pandas as pd
countries = {
    "BE": "Belgium",
    "CZ": "Czechia",
    "DK": "Denmark",
    "FR": "France",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "PL": "Poland",
    "RO": "Romania",
}

for code, name in countries.items():
    in_path = f"./dataset/simulated_{name}_ILI.csv"
    out_path = f"./dataset/simulated_{name}_ILI_median.csv"

    df = pd.read_csv(in_path)

    # time keys
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        time_keys = ["timestamp"]
    elif ("anno" in df.columns) and ("settimana" in df.columns):
        df["anno"] = df["anno"].astype(int)
        df["settimana"] = df["settimana"].astype(int)
        time_keys = ["anno", "settimana"]
    else:
        raise ValueError(f"{name}: Need timestamp or (anno,settimana) columns.")

    # median target column
    num_cols = ["incidenza"]

    # median per season per time
    gcols = ["season_id"] + time_keys
    df_med = df.groupby(gcols, as_index=False)[num_cols].median()

    # force item_id = 0
    df_med["item_id"] = 0

    # sort + save
    df_med = df_med.sort_values(["season_id"] + time_keys).reset_index(drop=True)
    df_med = df_med[["item_id", "season_id"] + time_keys + num_cols]
    df_med.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")