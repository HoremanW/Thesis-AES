# time_difference.py
import numpy as np
import pandas as pd

def estimate_delay_seconds_robust(
    time1, y1, time2, y2,
    max_lag_s=None,
    detrend_window_s=None,
    min_grid_step_s=0.5,
    max_grid_step_s=5.0
):
    """
    Lag > 0 means y2 (downhole) LAGS y1 (surface) by lag_s seconds.
    Uses a GLOBAL time origin, overlap-normalized NCC, optional detrend.
    """
    # --- Clean inputs ---
    y1 = pd.Series(y1, dtype="float64")
    y2 = pd.Series(y2, dtype="float64")
    t1 = pd.to_datetime(pd.Series(time1))
    t2 = pd.to_datetime(pd.Series(time2))

    m1 = y1.notna() & t1.notna()
    m2 = y2.notna() & t2.notna()
    y1, t1 = y1[m1].reset_index(drop=True), t1[m1].reset_index(drop=True)
    y2, t2 = y2[m2].reset_index(drop=True), t2[m2].reset_index(drop=True)

    if len(t1) < 2 or len(t2) < 2:
        raise ValueError("Need at least two points in each series.")

    # --- Global clock (not per-series zeros) ---
    t0 = min(t1.iloc[0], t2.iloc[0])
    ts1 = (t1 - t0).dt.total_seconds().to_numpy()
    ts2 = (t2 - t0).dt.total_seconds().to_numpy()

    # Enforce strictly increasing times
    keep1 = np.concatenate(([True], np.diff(ts1) > 0))
    keep2 = np.concatenate(([True], np.diff(ts2) > 0))
    ts1, y1 = ts1[keep1], y1.to_numpy()[keep1]
    ts2, y2 = ts2[keep2], y2.to_numpy()[keep2]

    # --- Choose grid step from native cadences ---
    def median_dt(ts):
        return 1.0 if len(ts) < 2 else float(np.median(np.diff(ts)))
    step_s = np.clip(min(median_dt(ts1), median_dt(ts2)), min_grid_step_s, max_grid_step_s)

    # --- Build global grid over the UNION duration ---
    t_max = float(max(ts1[-1], ts2[-1]))
    n = int(np.floor(t_max / step_s))
    if n < 10:
        raise ValueError("Not enough duration to estimate delay.")
    t_grid = step_s * np.arange(n + 1)  # 0..t_max

    # --- Interpolate onto the global grid; outside native span -> NaN (not edge-filled) ---
    def interp_nan(ts, y, tg):
        yint = np.interp(tg, ts, y, left=np.nan, right=np.nan)
        # np.interp won't emit NaN by itself; implement NaN outside convex hull:
        yint[tg < ts[0]] = np.nan
        yint[tg > ts[-1]] = np.nan
        return yint

    y1i = interp_nan(ts1, y1, t_grid)
    y2i = interp_nan(ts2, y2, t_grid)

    # --- Optional detrend with rolling mean (ignore NaNs) ---
    if detrend_window_s and detrend_window_s > step_s:
        win = int(round(detrend_window_s / step_s))
        win = max(3, win | 1)
        def detrend(x):
            s = pd.Series(x)
            return (s - s.rolling(win, center=True, min_periods=1).mean()).to_numpy()
        y1i, y2i = detrend(y1i), detrend(y2i)

    # --- Z-score each valid overlap at each lag; pick lag maximizing NCC ---
    def zscore(x):
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        return (x - mu) / (sd if sd > 0 else 1.0)

    # Pre-z for full vectors; we will re-z per-overlap slice to be robust
    # Bounds for lag indices in samples
    max_k = len(t_grid) - 1
    if max_lag_s is not None:
        max_k = min(max_k, int(abs(max_lag_s) / step_s))

    best_lag_idx = 0
    best_ncc = -np.inf

    for k in range(-max_k, max_k + 1):
        if k >= 0:
            a = y1i[k:]
            b = y2i[:len(y1i)-k]
        else:
            a = y1i[:len(y1i)+k]
            b = y2i[-k:]

        # valid overlap (both non-NaN)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            continue

        az = zscore(a[m])
        bz = zscore(b[m])
        # normalized cross-correlation = mean(az * bz)
        ncc = float(np.mean(az * bz))

        if ncc > best_ncc:
            best_ncc = ncc
            best_lag_idx = k

    lag_s = best_lag_idx * step_s
    return float(lag_s), float(step_s)
