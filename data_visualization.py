import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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