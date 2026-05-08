import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io

EXCLUDED_METHODS = {
    "ensembleun",
    "ensemble_un",
    "ensemble-un",
    "unweighted_ensemble",
    "unweighted ensemble",
}

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("./results/metrics_tables/point_metrics_long_real_sim.csv")

# keep only real-data rows
df = df[df["dataset_type"] == "real"].copy()
method_key = df["method"].astype(str).str.strip().str.lower()
df = df[~method_key.isin(EXCLUDED_METHODS)].copy()

# build a model label: method + train_setting (if non-empty)
df["model"] = df.apply(
    lambda r: r["method"] if (pd.isna(r["train_setting"]) or r["train_setting"] == "")
              else f"{r['method']}_{r['train_setting']}",
    axis=1,
)

# ── helper: count wins per model per step ─────────────────────────────────────
def count_wins(df, metric_name):
    """
    For each (step, country) pair pick the model with the lowest metric value.
    Return a DataFrame with columns: model, step, wins.
    """
    sub = df[df["metric"] == metric_name].copy()
    records = []
    for step in [1, 2, 3, 4]:
        s = sub[sub["step"] == step]
        for country, grp in s.groupby("country"):
            if grp.empty:
                continue
            winner = grp.loc[grp["value"].idxmin(), "model"]
            records.append({"model": winner, "step": step})
    won = pd.DataFrame(records)
    pivot = (
        won.groupby(["model", "step"])
        .size()
        .reset_index(name="wins")
        .pivot(index="model", columns="step", values="wins")
        .fillna(0)
        .astype(int)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
    )
    return pivot

# ── build win tables ──────────────────────────────────────────────────────────
mae_wins = count_wins(df, "MAE")
wis_wins = count_wins(df, "WIS80_mean")

# ── desired model order (readable labels) ────────────────────────────────────
def sort_by_total_wins(wins_df):
    """Sort models by total wins descending (most wins on the left)."""
    wins_df = wins_df.copy()
    wins_df["_total"] = wins_df.sum(axis=1)
    wins_df = wins_df.sort_values("_total", ascending=False).drop(columns="_total")
    return wins_df

mae_wins = sort_by_total_wins(mae_wins)
wis_wins = sort_by_total_wins(wis_wins)

# rename display labels
DISPLAY_NAMES = {
    "TabPFN_ts": "TabPFN-TS",
}
SETTING_LABELS = {
    "combined": "comb",
    "augmented": "aug",
}

def display_model_label(model_name):
    """Convert internal names such as Autoformer_combined to plot labels."""
    model_name = str(model_name)
    for setting, short in SETTING_LABELS.items():
        suffix = f"_{setting}"
        if model_name.endswith(suffix):
            base = model_name[:-len(suffix)]
            return f"{DISPLAY_NAMES.get(base, base)} ({short})"
    return DISPLAY_NAMES.get(model_name, model_name)

mae_wins.index = mae_wins.index.map(display_model_label)
wis_wins.index = wis_wins.index.map(display_model_label)

# ── plot ──────────────────────────────────────────────────────────────────────
# 使用高级的非饱和/莫兰迪色系 (Muted/Pastel)
STEP_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
STEP_LABELS = ["Step 1", "Step 2", "Step 3", "Step 4"]

def make_bar_chart(wins_df, title, ax, show_legend=True):
    n_models = len(wins_df)
    n_steps  = 4
    bar_w    = 0.13
    x        = np.arange(n_models)

    offsets = np.linspace(-(n_steps - 1) / 2, (n_steps - 1) / 2, n_steps) * bar_w

    for i, step in enumerate([1, 2, 3, 4]):
        vals = wins_df[step].values
        ax.bar(x + offsets[i], vals, width=bar_w, color=STEP_COLORS[i],
               label=STEP_LABELS[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(wins_df.index, rotation=0, ha="center", fontsize=16)
    ax.set_ylabel("Number of country-wins", fontsize=17)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.set_ylim(0, 9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    # 只有当 show_legend 为 True 时才绘制图例
    if show_legend:
        ax.legend(title="Forecast horizon", fontsize=15, title_fontsize=16)
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.6, n_models - 0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 去掉了 fig.suptitle，并为下方第一张图开启图例，第二张图关闭图例
make_bar_chart(mae_wins, "MAE — model wins per forecast step", axes[0], show_legend=True)
make_bar_chart(wis_wins, "IS₈₀ — model wins per forecast step", axes[1], show_legend=False)

for ax, label in zip(axes, ["(a)", "(b)"]):
    ax.text(-0.07, 1.05, label, transform=ax.transAxes,
            fontsize=19, fontweight="bold", va="top", ha="left")

plt.tight_layout()
plt.savefig("./test_results/montages/win_counts_MAE_WIS.png", dpi=300, bbox_inches="tight")
print("Saved: win_counts_MAE_WIS.png")
