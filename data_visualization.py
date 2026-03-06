import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_single_region_boxplot(base_dir, horizon, settings, save_path=None, dpi=300):
    # horizon 必须是 1,2,3...
    h = int(horizon)

    methods = ['ARIMA', 'DLinear', 'LSTM', 'Autoformer', 'TabPFN']
    label_map = {
        'ARIMA': 'ARIMA',
        'DLinear': 'DLinear',
        'LSTM': 'LSTM',
        'Autoformer': 'Autoformer',
        'TabPFN': 'TabPFN-TS',
    }

    baseline_path = os.path.join(base_dir, f"{settings}_Italy_Naive", f"wis80_point_step{h}.npy")
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file not found at {baseline_path}")
        return

    baseline_wis = np.load(baseline_path)
    baseline_wis = np.where(baseline_wis == 0, 1e-9, baseline_wis)

    all_rel_data, final_labels = [], []

    for m in methods:
        file_path = os.path.join(base_dir, f"{settings}_Italy_{m}", f"wis80_point_step{h}.npy")
        if not os.path.exists(file_path):
            print(f"File missing: {file_path}")
            continue

        method_wis = np.load(file_path)
        relative_wis = method_wis / baseline_wis

        all_rel_data.append(relative_wis)
        final_labels.append(label_map[m])

    if len(all_rel_data) == 0:
        print("No method files were found, nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    # --- 修改处 1: showfliers 改为 True，并定义离群点样式 (flierprops) ---
    flier_style = dict(marker='o', markerfacecolor='gray', markersize=3,
                      linestyle='none', markeredgecolor='none', alpha=0.4)
    bp = ax.boxplot(all_rel_data, tick_labels=final_labels, showfliers=True, patch_artist=True, flierprops=flier_style)

    for i, data_array in enumerate(all_rel_data):
        median_val = np.median(data_array)
        target_color = 'green' if median_val < 1 else 'orange'
        bp['boxes'][i].set(facecolor='#f0f2f5', edgecolor='#444444', linewidth=1.5)
        bp['medians'][i].set_color(target_color)
        bp['medians'][i].set_linewidth(3)

    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel('Relative WIS (vs Naive)', fontsize=12, fontweight='bold')
    ax.set_title(f'{settings} Italy Data: Relative WIS Comparison ({h}-step-ahead)', fontsize=14, pad=20)

    plt.xticks(rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='green', lw=3),
        Line2D([0], [0], color='orange', lw=3),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--')
    ]
    ax.legend(custom_lines, ['Better than Naive', 'Worse than Naive', 'Baseline'], loc='upper right')

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()
if __name__ == "__main__":
    in_path = "dataset/simulated_Italy_ILI_item0.csv"
    df = pd.read_csv(in_path)

    # ---------- identify ID column (item_id or series_id) ----------
    if "item_id" in df.columns:
        id_col = "item_id"
    elif "series_id" in df.columns:
        id_col = "series_id"
    else:
        raise ValueError("Expected an 'item_id' or 'series_id' column in the CSV.")

    # ---------- identify year and week columns ----------
    # Adjust these if your column names are slightly different
    if "year" in df.columns:
        year_col = "year"
    elif "anno" in df.columns:
        year_col = "anno"
    else:
        raise ValueError("Could not find a year column (looked for 'year' or 'anno').")

    if "settimana" in df.columns:
        week_col = "settimana"
    elif "week" in df.columns:
        week_col = "week"
    else:
        raise ValueError("Could not find a week column (looked for 'settimana' or 'week').")

    # ---------- build a proper datetime from (year, settimana) ----------
    # We interpret (year, settimana) as ISO week: Monday of that week
    week_str = df[week_col].astype(int).astype(str).str.zfill(2)
    df["date"] = pd.to_datetime(
        df[year_col].astype(int).astype(str) + "-W" + week_str + "-1",
        format="%G-W%V-%u"
    )

    # sort just to be safe
    df = df.sort_values([id_col, "date"])

    # ---------- prepare IDs and date range ----------
    ids = sorted(df[id_col].unique())  # should be 0..9
    x_min = df["date"].min()
    x_max = df["date"].max()

    fig, ax = plt.subplots(figsize=(16, 6))

    # use a categorical colormap with distinct colors
    cmap = plt.get_cmap("tab10")  # good for up to 10 series

    for i, this_id in enumerate(ids):
        sub = df[df[id_col] == this_id]

        ax.plot(
            sub["date"],
            sub["incidenza"],
            linewidth=1.5,
            color=cmap(i % cmap.N),
            label=f"{id_col} = {this_id}",
        )

    # x range shared for all
    ax.set_xlim(x_min, x_max)

    # labels and title
    ax.set_title("China ILI incidence time series (2016–2025) – item_id 0–9", fontsize=16)
    ax.set_xlabel("Year / settimana (ISO week)")
    ax.set_ylabel("incidenza")

    # nicer x tick labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # legend to distinguish the series
    ax.legend(title=id_col, ncol=2, fontsize=8)

    fig.tight_layout()
    plt.savefig('dataset/simulated_Italy_ILI_test.png',
                dpi=300, bbox_inches="tight")
    plt.show()