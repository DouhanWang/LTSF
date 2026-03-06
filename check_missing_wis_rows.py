import pandas as pd

IN_PATH = "dataset/latest-forecast_scores.csv"
START_ORIGIN = "2024-11-17"
END_ORIGIN = "2025-04-09"

TEAM_ID = "respicast"
MODEL_KEYWORD = "hubensemble"
TARGET_KEYWORD = "ILI"
INTERVAL_METRIC = "WIS"

COUNTRIES = ["BE","CZ","DK","FR","IE","IT","NL","PL","RO"]

def parse_origin_date(s):
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.fillna(dt2)

df = pd.read_csv(IN_PATH)
for c in ["team_id","model_id","metric","target","location","origin_date"]:
    df[c] = df[c].astype(str).str.strip()

df["origin_date_dt"] = parse_origin_date(df["origin_date"])
df = df[(df["origin_date_dt"] >= pd.to_datetime(START_ORIGIN)) &
        (df["origin_date_dt"] <= pd.to_datetime(END_ORIGIN))]

df = df[
    (df["team_id"].str.lower() == TEAM_ID.lower()) &
    (df["model_id"].str.lower().str.contains(MODEL_KEYWORD, na=False)) &
    (df["target"].str.contains(TARGET_KEYWORD, case=False, na=False)) &
    (df["metric"].str.upper() == INTERVAL_METRIC)
].copy()

# 每国每step应该 21 个
expected = df.groupby(["location","horizon"])["origin_date_dt"].nunique().unstack(fill_value=0)
print("Counts of unique origin_date per (country, horizon):")
print(expected.loc[COUNTRIES, [1,2,3,4]])

# 直接列出 France step3/4 缺的是哪些 origin_date
for loc in ["FR","RO"]:
    for h in [1,2,3,4]:
        sub = df[(df["location"]==loc) & (df["horizon"].astype(int)==h)].copy()
        dates = sorted(sub["origin_date_dt"].dt.date.unique())
        print(f"\n{loc} horizon {h}: n={len(dates)}")
        print(dates)