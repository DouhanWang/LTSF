import pandas as pd
import numpy as np

def shift_with_reflect(a: np.ndarray, k: int) -> np.ndarray:
    """
    Shift 1D array by k steps without wrap-around.
    +k: shift right (later), fill left gap by reflecting early values
    -k: shift left (earlier), fill right gap by reflecting late values
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    if k == 0 or n == 0:
        return a.copy()

    if k > 0:
        k = min(k, n)
        if n > 1:
            # reflect from the start: a[1], a[2], ... mirrored
            pad = a[1:k+1][::-1] if k < n else a[1:][::-1]
            if len(pad) < k:
                pad = np.pad(pad, (0, k - len(pad)), mode="edge")
        else:
            pad = np.repeat(a[0], k)
        out = np.concatenate([pad, a[:-k]])
    else:
        k = min(-k, n)
        if n > 1:
            # reflect from the end
            pad = a[-k-1:-1][::-1] if k < n else a[:-1][::-1]
            if len(pad) < k:
                pad = np.pad(pad, (0, k - len(pad)), mode="edge")
        else:
            pad = np.repeat(a[-1], k)
        out = np.concatenate([a[k:], pad])

    return np.clip(out, 0.0, None)


def add_realistic_noise(a: np.ndarray, rng: np.random.Generator,
                        mult_range=(0.03, 0.12), add_frac=0.02) -> np.ndarray:
    """Heteroskedastic multiplicative noise + small additive noise."""
    a = np.asarray(a, dtype=float)
    season_std = np.nanstd(a) if np.isfinite(a).any() else 0.0

    mult = rng.uniform(*mult_range)  # noise strength varies per-aug sample
    eps_mult = rng.normal(0.0, mult, size=a.shape) * np.maximum(a, 1.0)
    eps_add = rng.normal(0.0, add_frac * season_std, size=a.shape)

    out = a + eps_mult + eps_add
    out = np.clip(out, 0.0, None)
    return out

def maybe_smooth(a: np.ndarray, rng: np.random.Generator, p=0.30) -> np.ndarray:
    """Optional light smoothing to keep epidemic-like shape."""
    if rng.random() > p:
        return a
    w = int(rng.integers(3, 6))  # 3-5
    kernel = np.ones(w) / w
    padded = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def augment_italy_csv(
    in_csv: str,
    out_csv: str = "augmented_Italy.csv",
    n_aug: int = 1000,
    seed: int = 20260218,
    shift_range=(-4, 4),
    force_nonzero_shift: bool = True,
    mult_range=(0.03, 0.12),
    add_frac: float = 0.02,
    smooth_p: float = 0.30,
):
    """
    Input CSV must contain columns:
      item_id, season_id, anno, settimana, incidenza
    Real data must be item_id=0.
    Output:
      item_id=0 is real, item_id=1..n_aug are augmented (shift peak + noise).
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(in_csv)

    required = {"item_id", "season_id", "anno", "settimana", "incidenza"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

    base = df[df["item_id"] == 0].copy()
    if base.empty:
        raise ValueError("No rows with item_id=0 found (real series expected).")

    base = base.sort_values(["season_id", "anno", "settimana"]).reset_index(drop=True)
    seasons = sorted(base["season_id"].unique())

    # pre-split by season
    season_blocks = {
        s: base[base["season_id"] == s].sort_values(["anno", "settimana"]).reset_index(drop=True)
        for s in seasons
    }

    rows = [base.copy()]  # keep real (item_id=0)

    for aug_id in range(1, n_aug + 1):
        parts = []
        for s in seasons:
            blk = season_blocks[s].copy()
            y = blk["incidenza"].to_numpy(dtype=float)

            k = int(rng.integers(shift_range[0], shift_range[1] + 1))
            if force_nonzero_shift and k == 0:
                k = int(rng.integers(shift_range[0], shift_range[1] + 1))

            y_shift = shift_with_reflect(y, k)
            y_noisy = add_realistic_noise(y_shift, rng, mult_range=mult_range, add_frac=add_frac)

            # override smoothing probability if you want
            if smooth_p is not None:
                # re-implement with custom p
                if rng.random() <= smooth_p:
                    w = int(rng.integers(3, 6))
                    kernel = np.ones(w) / w
                    padded = np.pad(y_noisy, (w // 2, w - 1 - w // 2), mode="edge")
                    y_noisy = np.convolve(padded, kernel, mode="valid")

            blk["incidenza"] = y_noisy
            blk["item_id"] = aug_id
            parts.append(blk)

        rows.append(pd.concat(parts, ignore_index=True))

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["item_id", "season_id", "anno", "settimana"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} | series: {out['item_id'].nunique()} | rows: {len(out)}")

if __name__ == "__main__":
    augment_italy_csv(
        in_csv="./dataset/per_country_csv/RO.csv",                 # <- your real file (item_id=0)
        out_csv="./dataset/augmented_Romania_ILI.csv",    # <- output
        n_aug=1000,
        shift_range=(-4, 4),
        mult_range=(0.03, 0.12),
        add_frac=0.02,
        smooth_p=0.30,
        seed=20260218,
    )
