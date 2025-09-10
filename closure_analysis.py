import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def find_shut_in_time_from_flow(time_surface, flow_m3h, threshold=0.1, min_hold_s=30):
    """
    Heuristic: shut-in is the first timestamp after the max flow where
    flow drops below `threshold` and stays there for `min_hold_s`.
    """
    t = pd.to_datetime(pd.Series(time_surface)).reset_index(drop=True)
    q = pd.Series(flow_m3h, dtype=float).reset_index(drop=True)
    if len(q.dropna()) == 0:
        return None
    imax = int(q.idxmax())
    # Look forward from peak
    for i in range(imax, len(q)):
        if pd.notna(q.iloc[i]) and q.iloc[i] <= threshold:
            t0 = t.iloc[i]
            # hold check
            j = i
            ok = True
            while j < len(q) and (t.iloc[j] - t0).total_seconds() <= min_hold_s:
                if pd.notna(q.iloc[j]) and q.iloc[j] > threshold:
                    ok = False
                    break
                j += 1
            if ok:
                return t0
    return None

def build_shut_in_series(time, pressure, t_shut_in):
    """
    Returns (t_s, p) where t_s is seconds since shut-in (np.ndarray),
    trimmed to t_s >= 0, with NaNs dropped and time strictly increasing.
    """
    t = pd.to_datetime(pd.Series(time))
    p = pd.Series(pressure, dtype=float)
    m = t.notna() & p.notna()
    t, p = t[m].reset_index(drop=True), p[m].reset_index(drop=True)
    ts = (t - pd.to_datetime(t_shut_in)).dt.total_seconds().to_numpy()
    keep = ts >= 0
    ts, p = ts[keep], p.to_numpy()[keep]
    # enforce strictly increasing
    if len(ts) >= 2:
        keep2 = np.concatenate(([True], np.diff(ts) > 0))
        ts, p = ts[keep2], p[keep2]
    return ts, p

def bourdet_derivative(t_seconds, p, smooth_win=None):
    """
    Bourdet (log-log) derivative on irregular sampling.
    Returns (t_mid, dp_dlogt).
    If smooth_win (odd int) is given, apply a moving median smoother to p first.
    """
    t = np.asarray(t_seconds, float)
    p = np.asarray(p, float)
    if smooth_win and smooth_win >= 3 and (smooth_win % 2 == 1):
        from scipy.signal import medfilt
        p = medfilt(p, kernel_size=smooth_win)

    # Need at least 3 points
    if len(t) < 3:
        return t, np.full_like(t, np.nan)

    ln_t = np.log(t, where=t>0, out=np.full_like(t, np.nan))
    # central Bourdet derivative
    dp = np.empty_like(p)
    dp[:] = np.nan
    for i in range(1, len(t)-1):
        i0, i2 = i-1, i+1
        denom = ln_t[i2] - ln_t[i0]
        if np.isfinite(denom) and denom != 0:
            dp[i] = ((p[i] - p[i0])/(ln_t[i] - ln_t[i0]) + (p[i2] - p[i])/(ln_t[i2] - ln_t[i])) / 2.0
    # We return t (seconds since shut-in) aligned to p; dp is dP/d(ln t).
    return t, dp

def derivative_vs_sqrt_time(t_seconds, p):
    """
    Returns sqrt(t), P, and numerical derivative dP/d(sqrt(t)).
    """
    t = np.asarray(t_seconds, float)
    p = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p) & (t >= 0)
    t, p = t[m], p[m]
    if len(t) < 3:
        return np.sqrt(t), p, np.full_like(t, np.nan)
    sqrt_t = np.sqrt(t)
    # derivative wrt sqrt(t): dP/dx where x = sqrt(t)
    # Use central differences in x
    x = sqrt_t
    dpdx = np.empty_like(p)
    dpdx[:] = np.nan
    for i in range(1, len(x)-1):
        dx = x[i+1] - x[i-1]
        if dx != 0:
            dpdx[i] = (p[i+1] - p[i-1]) / dx
    return x, p, dpdx

def suggest_closure_from_srt(x_sqrt_t, p, dpdx, min_t_s=60, guard_s=5):
    """
    Very simple heuristic:
    - Ignore the first `min_t_s` seconds (early transients).
    - Find the first pronounced *upturn* in dP/d√t (local minimum).
    Returns index of suggested closure (or None).
    """
    if len(x_sqrt_t) < 5:
        return None
    t = (x_sqrt_t**2)
    m = t >= min_t_s
    idx = np.where(m)[0]
    if len(idx) < 5:
        return None
    i0 = idx[0]
    # local minima in dpdx after i0
    cand = None
    best_val = np.inf
    for i in range(i0+guard_s, len(dpdx)-guard_s):
        if np.isfinite(dpdx[i]):
            if dpdx[i-1] > dpdx[i] < dpdx[i+1]:  # local min
                if dpdx[i] < best_val:
                    best_val = dpdx[i]
                    cand = i
    return cand


