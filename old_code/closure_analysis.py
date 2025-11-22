import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def find_shut_in_time_from_flow(time_surface, flow_m3h, threshold=0.1, min_hold_s=30):
    """
    Infer the shut-in timestamp from a surface flow signal.

    Heuristic
    ---------
    1) Find the time of maximum measured flow (assumed during pumping).
    2) From that point forward, the first time the flow drops to `<= threshold`
       and *stays* at or below that threshold for at least `min_hold_s`
       seconds is labeled as the shut-in time.

    Parameters
    ----------
    time_surface : array-like (datetime64-like or strings convertible by pandas)
        Timestamps for the surface flow measurements.
    flow_m3h : array-like of float
        Flow rate in m^3/h at `time_surface`. NaNs are ignored.
    threshold : float, optional (default 0.1)
        Flow cutoff considered “zero flow” / shut-in, in m^3/h.
        Pick a small positive value to be robust to sensor noise.
    min_hold_s : float, optional (default 30)
        Required duration (in seconds) the flow must remain at or below
        `threshold` to confirm shut-in.

    Returns
    -------
    pandas.Timestamp or None
        Estimated shut-in timestamp if found, otherwise None.

    Notes
    -----
    - This is a conservative detector that avoids transient dips by
      requiring a hold time.
    - Assumes roughly monotonic reduction of flow after the peak.
    - If the signal is very noisy, consider smoothing flow before calling.
    """
    t = pd.to_datetime(pd.Series(time_surface)).reset_index(drop=True)
    q = pd.Series(flow_m3h, dtype=float).reset_index(drop=True)

    if len(q.dropna()) == 0:
        return None

    imax = int(q.idxmax())  # start looking after pumping peak

    # Scan forward to find first low-flow sample that holds low for min_hold_s
    for i in range(imax, len(q)):
        if pd.notna(q.iloc[i]) and q.iloc[i] <= threshold:
            t0 = t.iloc[i]
            # Verify the low-flow condition persists for min_hold_s
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


def build_shut_in_series(time, pressure, t_shut_in, max_seconds=None, insert_isip=True):
    """
    Build (t_s, p) since shut-in. Optionally insert an ISIP anchor at t=0
    by linear interpolation between the last pre-shut sample and first post-shut sample.
    """
    # Keep pandas Series for safe positional indexing
    t_series = pd.to_datetime(pd.Series(time))
    p_series = pd.Series(pressure, dtype=float)

    # Drop NaNs consistently
    m = t_series.notna() & p_series.notna()
    t_clean = t_series[m].reset_index(drop=True)
    p_clean = p_series[m].reset_index(drop=True)

    # Time since shut-in (in seconds) for the cleaned arrays
    t0 = pd.to_datetime(t_shut_in)
    ts_all = (t_clean - t0).dt.total_seconds()

    # Keep only t >= 0
    keep_nonneg = ts_all >= 0
    ts = ts_all[keep_nonneg].to_numpy()
    p  = p_clean[keep_nonneg].to_numpy()

    # Enforce strictly increasing time (remove zero-time duplicates)
    if len(ts) >= 2:
        k2 = np.concatenate(([True], np.diff(ts) > 0))
        ts, p = ts[k2], p[k2]

    # Optional ISIP anchor at exactly t=0 if first kept sample is after shut-in
    if insert_isip and (len(ts) == 0 or ts[0] > 0):
        # Find neighbors around t0 in the CLEANED (NaN-free) arrays, *not* the kept subset
        i_before = np.where(t_clean <= t0)[0]
        i_after  = np.where(t_clean >= t0)[0]

        if len(i_before) and len(i_after):
            i0, i1 = int(i_before[-1]), int(i_after[0])
            tL, tR = t_clean.iloc[i0].value, t_clean.iloc[i1].value
            pL, pR = p_clean.iloc[i0],      p_clean.iloc[i1]
            if np.isfinite(pL) and np.isfinite(pR) and tR != tL:
                w = (t0.value - tL) / (tR - tL)
                p0 = (1.0 - w) * pL + w * pR
            else:
                # fallback: if interpolation is impossible, use nearest available value
                p0 = pL if np.isfinite(pL) else (pR if np.isfinite(pR) else (p[0] if len(p) else np.nan))
        else:
            # If we cannot bracket t0, fall back to first kept sample (if any)
            p0 = p[0] if len(p) else (p_clean.iloc[0] if len(p_clean) else np.nan)

        # Prepend the anchor
        ts = np.insert(ts, 0, 0.0)
        p  = np.insert(p,  0, float(p0))

    # Hard cap by max_seconds if requested
    if max_seconds is not None and np.isfinite(max_seconds):
        k3 = ts <= float(max_seconds)
        ts, p = ts[k3], p[k3]

    return ts, p

import numpy as np

def _two_segment_intersection(x, y, min_left=8, min_right=8):
    """
    Fit two lines (left/right) separated by a breakpoint to (x, y),
    choose the breakpoint that minimizes total SSE, and return the
    intersection point and details.

    Returns
    -------
    result : dict
      {
        'ok': bool,
        'i_break': int,     # index where right segment starts
        'a1': float, 'b1': float,   # left line y = a1*x + b1
        'a2': float, 'b2': float,   # right line y = a2*x + b2
        'x_star': float, 'y_star': float,   # intersection
        'sse': float
      }
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    n = len(x)
    if n < (min_left + min_right + 1):
        return {'ok': False, 'reason': 'not_enough_points'}

    # precompute for speed (simple polyfit on each side per candidate)
    best = {'sse': np.inf, 'ok': False}
    # candidate breakpoint i_break means: left = [0:i_break), right=[i_break:n)
    for i_break in range(min_left, n - min_right):
        xl, yl = x[:i_break], y[:i_break]
        xr, yr = x[i_break:], y[i_break:]

        # guard degenerate spans
        if np.allclose(xl.ptp(), 0) or np.allclose(xr.ptp(), 0):
            continue

        a1, b1 = np.polyfit(xl, yl, 1)
        a2, b2 = np.polyfit(xr, yr, 1)

        yhat_l = a1*xl + b1
        yhat_r = a2*xr + b2
        sse = np.nansum((yl - yhat_l)**2) + np.nansum((yr - yhat_r)**2)

        if sse < best['sse']:
            best = dict(ok=True, i_break=i_break, a1=a1, b1=b1, a2=a2, b2=b2, sse=sse)

    if not best['ok']:
        return best

    # intersection of a1*x + b1 = a2*x + b2
    denom = (best['a1'] - best['a2'])
    if np.isclose(denom, 0.0):
        # nearly parallel; take the x at breakpoint as fallback
        x_star = x[best['i_break']]
        y_star = best['a1']*x_star + best['b1']
    else:
        x_star = (best['b2'] - best['b1']) / denom
        y_star = best['a1']*x_star + best['b1']

    best['x_star'] = float(x_star)
    best['y_star'] = float(y_star)
    return best


def bourdet_derivative(t_seconds, p, smooth_win=None, max_t_s=None, min_ln_spacing=1e-6):
    """
    Bourdet derivative dP/d(ln t) with small-Δln t guard and robust smoothing.

    Parameters
    ----------
    t_seconds : array-like
        Seconds since shut-in (> 0 recommended).
    p : array-like
        Pressure [bar] aligned to t_seconds.
    smooth_win : int or None
        Odd kernel size (>=3) for median filter on P before differencing.
    max_t_s : float or None
        If set, cap analysis to t <= max_t_s.
    min_ln_spacing : float
        Minimum |Δ ln t| to accept for central differencing (avoid tiny gaps).

    Returns
    -------
    t : np.ndarray
        Time array (possibly capped).
    dP_dln_t : np.ndarray
        Bourdet derivative; NaN at endpoints or where not computable.
    """
    t  = np.asarray(t_seconds, dtype=np.float64)
    pp = np.asarray(p,          dtype=np.float64)

    # Cap window first (keeps plots/analysis consistent)
    if max_t_s is not None:
        mcap = np.isfinite(t) & (t <= float(max_t_s))
        t, pp = t[mcap], pp[mcap]

    # Keep only finite & strictly positive time
    m = np.isfinite(t) & np.isfinite(pp) & (t > 0)
    t, pp = t[m], pp[m]
    if len(t) < 3:
        return t, np.full_like(t, np.nan)

    # OPTIONAL smoothing of P (median filter)
    if smooth_win and smooth_win >= 3 and (smooth_win % 2 == 1):
        try:
            from scipy.signal import medfilt
            pp = medfilt(pp, kernel_size=int(smooth_win))
        except Exception:
            pass  # no SciPy → just continue

    # ln t with NaN for any non-positive t (already filtered)
    ln_t = np.log(t)

    d = np.full_like(pp, np.nan)
    # Central Bourdet derivative
    for i in range(1, len(t) - 1):
        i0, i2 = i - 1, i + 1
        dl_left  = ln_t[i]  - ln_t[i0]
        dl_right = ln_t[i2] - ln_t[i]
        dl_span  = ln_t[i2] - ln_t[i0]

        # Guard against tiny ln spacings
        if (abs(dl_span)  < min_ln_spacing or
            abs(dl_left)  < min_ln_spacing or
            abs(dl_right) < min_ln_spacing):
            continue

        left  = (pp[i]  - pp[i0]) / dl_left
        right = (pp[i2] - pp[i])  / dl_right
        d[i]  = 0.5 * (left + right)

    return t, d

def suggest_closure_from_bourdet(t_seconds, dp_dlogt, min_t_s=None, max_t_s=None):
    """
    Closure index = global minimum of Bourdet derivative dP/d(ln t)
    within [min_t_s, max_t_s].
    """

    t = np.asarray(t_seconds, float)
    d = np.asarray(dp_dlogt,  float)

    if len(t) < 3:
        return None

    mask = np.isfinite(t) & np.isfinite(d) & (t > 0)
    if min_t_s is not None:
        mask &= (t >= float(min_t_s))
    if max_t_s is not None:
        mask &= (t <= float(max_t_s))

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    i_min = idx[np.nanargmin(d[idx])]
    return int(i_min)

def derivative_vs_sqrt_time(t_seconds, p, max_t_s=None):
    """
    Compute pressure derivative with respect to sqrt(time): dP/d(√t).

    Useful for classical Nolte/Barree diagnostics where linear flow/closure
    manifests via trends in P vs √t and its slope.

    Parameters
    ----------
    t_seconds : array-like of float
        Time since shut-in in seconds (t >= 0 recommended).
    p : array-like of float
        Pressure values aligned to `t_seconds`.

    Returns
    -------
    sqrt_t : np.ndarray
        Square root of time (seconds^0.5).
    p : np.ndarray
        Pressure values aligned to `sqrt_t`.
    dpdx : np.ndarray
        Numerical central-difference derivative dP/d(√t); NaN at endpoints or
        where spacing is zero.

    Notes
    -----
    - Uses central differences in x = √t for interior points.
    - Ensure time is monotonic and non-negative for interpretable results.
    """
    t = np.asarray(t_seconds, float)
    pp = np.asarray(p, float)

    # keep finite and t >= 0; apply cap if requested
    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    t, pp = t[m], pp[m]

    n = len(t)
    if n < 2:
        # not enough points for any derivative
        return np.sqrt(t), pp, np.full_like(t, np.nan)
    if n == 2:
        # simple secant for both points
        x = np.sqrt(t)
        dx = x[1] - x[0]
        d = np.full_like(pp, np.nan)
        if dx != 0:
            val = (pp[1] - pp[0]) / dx
            d[0] = val
            d[1] = val
        return x, pp, d

    # general case (n >= 3)
    x = np.sqrt(t)
    d = np.empty_like(pp); d[:] = np.nan

    # central differences for interior points
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if dx != 0:
            d[i] = (pp[i + 1] - pp[i - 1]) / dx

    # one-sided at the ends
    dx0 = x[1] - x[0]
    if dx0 != 0:
        d[0] = (pp[1] - pp[0]) / dx0
    dxe = x[-1] - x[-2]
    if dxe != 0:
        d[-1] = (pp[-1] - pp[-2]) / dxe

    return x, pp, d

def suggest_closure_from_srt(x_sqrt_t, p, dpdx, min_t_s=None, guard_s=0, max_t_s=None):
    """
    Suggest a closure index from √t diagnostics by picking the global minimum of dP/d√t.

    Parameters
    ----------
    x_sqrt_t : array-like of float
        √t values (seconds^0.5), typically from `derivative_vs_sqrt_time`.
    p : array-like of float
        Pressure values at the same samples (not directly used here).
    dpdx : array-like of float
        dP/d(√t) from `derivative_vs_sqrt_time`.
    min_t_s : float, optional
        Minimum time since shut-in (seconds) before considering candidates.
    guard_s : int, optional (default 5)
        Guard band in samples to avoid picking very first/last points.
    max_t_s : float, optional
        Maximum time since shut-in (seconds) for consideration.

    Returns
    -------
    int or None
        Index of the minimum dP/d√t inside the window, or None if no valid candidate.
    """
    x_sqrt_t = np.asarray(x_sqrt_t, float)
    dpdx = np.asarray(dpdx, float)

    if len(x_sqrt_t) < 3:
        return None

    t = x_sqrt_t**2

    mask = np.isfinite(dpdx)
    if min_t_s is not None:
        mask &= (t >= float(min_t_s))
    if max_t_s is not None:
        mask &= (t <= float(max_t_s))

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    # Simply take the global minimum
    i_min = idx[np.nanargmin(dpdx[idx])]
    return int(i_min)



# --- Barree/Nolte high-efficiency G-function (Appendix A; low-leakoff assumption) ---
def g_function_high_efficiency(delta_t_seconds: np.ndarray, tp_seconds: float) -> np.ndarray:
    """
    Compute the high-efficiency (low-leakoff) G-function used in Barree/Nolte diagnostics.

    This matches the simplified Appendix A form commonly used for high
    fluid-efficiency (low-leakoff) cases.

    Parameters
    ----------
    delta_t_seconds : array-like of float
        Time since shut-in, Δt, in seconds.
    tp_seconds : float
        Pumping time, t_p, in seconds (normalizes Δt/t_p).

    Returns
    -------
    G : np.ndarray
        Dimensionless G-function values at each Δt/t_p.

    References
    ----------
    - Barree, R.D. (2009). *A Practical Guide to Hydraulic Fracture Diagnostic
      Interpretation*. Appendix A, eqs. (A-1)–(A-3).
    """
    d = np.asarray(delta_t_seconds, float) / max(1e-12, float(tp_seconds))
    root = np.sqrt(1.0 + d)
    # Intermediate g_D collapses to this closed form for the high-efficiency case.
    G = (4.0 / 3.0) * (root - 1.0) ** 2 + (root - 1.0)
    return G


def semilog_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute “semilog derivative” X * dY/dX (Barree plotting convention).

    Often used to visualize slopes on semilog-type axes without explicitly
    transforming the axis. Works for G, √t, Δt, and other x-axes used in
    closure diagnostics.

    Parameters
    ----------
    x : array-like of float
        Independent variable (e.g., G, √t, Δt, FL²). Should be monotonic.
    y : array-like of float
        Dependent variable (e.g., pressure or function of pressure).

    Returns
    -------
    out : np.ndarray
        Array of the same length as inputs with X * (dY/dX). Endpoints use
        one-sided differences; interior points use central differences.

    Notes
    -----
    - If x has repeated values (dx=0), derivative is undefined (NaN/inf).
    - This is a simple finite-difference operator; smooth noisy data beforehand
      for cleaner curves.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if len(x) < 3:
        return np.full_like(x, np.nan, dtype=float)

    dy = np.empty_like(y, dtype=float)
    dx = np.empty_like(x, dtype=float)

    # Central differences for interior points; one-sided at the ends
    dx[1:-1] = x[2:] - x[:-2]
    dy[1:-1] = y[2:] - y[:-2]
    dx[0]    = x[1] - x[0]
    dx[-1]   = x[-1] - x[-2]
    dy[0]    = y[1] - y[0]
    dy[-1]   = y[-1] - y[-2]

    with np.errstate(divide='ignore', invalid='ignore'):
        deriv = dy / dx

    return x * deriv


def _tp_seconds_from_cycle(cycle) -> float:
    """
    Robust pumping time (tp) in seconds for a cycle.
    tp = t_shut_in_surface - t_pump_start_surface, clipped to >=1 s.
    """
    try:
        tp = (cycle['t_shut_in_surface'] - cycle['t_pump_start_surface']).total_seconds()
    except Exception:
        tp = 0.0
    if not np.isfinite(tp) or tp <= 0:
        tp = 1.0
    return float(tp)

def suggest_closure_from_gfunction(ts_seconds, p, tp_seconds,
                                   *, min_t_s=None, max_t_s=None):
    """
    Closure index from G-function: global min of the semilog derivative of P vs G
    inside [min_t_s, max_t_s].
    """
    t = np.asarray(ts_seconds, float)
    pp = np.asarray(p, float)

    if len(t) < 3:
        return None

    # Build G and its semilog derivative curve
    G = g_function_high_efficiency(t, tp_seconds)
    semilog_dP = semilog_derivative(G, pp)

    # Windowing in *time* (seconds since shut-in), consistent with your other pickers
    mask = np.isfinite(t) & np.isfinite(semilog_dP)
    if min_t_s is not None:
        mask &= (t >= float(min_t_s))
    if max_t_s is not None:
        mask &= (t <= float(max_t_s))

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    i_min = idx[np.nanargmin(semilog_dP[idx])]
    return int(i_min)

def g_diagnostic(ts_seconds, p, tp_seconds):
    """Return (G, semilog_dP) for plotting/QC."""
    t = np.asarray(ts_seconds, float)
    pp = np.asarray(p, float)
    G = g_function_high_efficiency(t, tp_seconds)
    semilog_dP = semilog_derivative(G, pp)
    return G, semilog_dP


# --- √t intersection (returns line fits too) ---
def fcp_by_sqrt_intersection(ts_seconds, p, *, max_t_s=None, min_left=8, min_right=8):
    """
    Fit straight lines to early/late segments of P vs √t and return
    the intersection + line fits (for plotting).
    """
    t = np.asarray(ts_seconds, float)
    pp = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    if not m.any():
        return {"ok": False}

    x = np.sqrt(t[m])      # √t
    y = pp[m]

    if len(x) < (min_left + min_right):
        return {"ok": False}

    xL, yL = x[:min_left],      y[:min_left]
    xR, yR = x[-min_right:],    y[-min_right:]

    aL, bL = np.polyfit(xL, yL, 1)
    aR, bR = np.polyfit(xR, yR, 1)

    if np.isclose(aL, aR):
        return {"ok": False}

    x_int = (bR - bL) / (aL - aR)
    y_int = aL * x_int + bL

    # small helper segments to draw (exactly over the data range)
    xL_fit = np.array([xL.min(), xL.max()])
    yL_fit = aL * xL_fit + bL
    xR_fit = np.array([xR.min(), xR.max()])
    yR_fit = aR * xR_fit + bR

    return {
        "ok": True,
        "closure_time_s": float(x_int**2),
        "closure_pressure_bar": float(y_int),
        "aL": aL, "bL": bL, "aR": aR, "bR": bR,
        "xL_fit": xL_fit, "yL_fit": yL_fit,
        "xR_fit": xR_fit, "yR_fit": yR_fit,
        # for convenience if you want to re-plot data on √t
        "x_data": x, "y_data": y
    }


# --- G-function intersection (returns line fits too) ---
def fcp_by_g_intersection(ts_seconds, p, *, tp_seconds, max_t_s=None, min_left=8, min_right=8):
    """
    Fit straight lines to early/late segments of P vs G and return
    the intersection + line fits (for plotting).
    """
    t = np.asarray(ts_seconds, float)
    pp = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    if not m.any():
        return {"ok": False}

    G = g_function_high_efficiency(t[m], float(tp_seconds))
    y = pp[m]

    if len(G) < (min_left + min_right):
        return {"ok": False}

    xL, yL = G[:min_left],   y[:min_left]
    xR, yR = G[-min_right:], y[-min_right:]

    aL, bL = np.polyfit(xL, yL, 1)
    aR, bR = np.polyfit(xR, yR, 1)

    if np.isclose(aL, aR):
        return {"ok": False}

    x_int = (bR - bL) / (aL - aR)
    y_int = aL * x_int + bL

    xL_fit = np.array([xL.min(), xL.max()])
    yL_fit = aL * xL_fit + bL
    xR_fit = np.array([xR.min(), xR.max()])
    yR_fit = aR * xR_fit + bR

    # We keep closure_time_s in seconds (your main time base). If you only need the
    # intersection on G for plotting, it's x_int (dimensionless); time mapping not needed here.
    return {
        "ok": True,
        "closure_time_s": float((x_int - G[0]) / (G[-1] - G[0]) * (t[m][-1] - t[m][0]) + t[m][0]),
        "closure_pressure_bar": float(y_int),
        "aL": aL, "bL": bL, "aR": aR, "bR": bR,
        "xL_fit": xL_fit, "yL_fit": yL_fit,
        "xR_fit": xR_fit, "yR_fit": yR_fit,
        "x_int_g": float(x_int),   # handy for plotting the vertical marker on G
        "x_data": G, "y_data": y
    }



def analyze_all_shutins(
    cycles,
    *,
    time_S, flowrate_S, return_volume_S,
    time_D, p_downhole_corr,
    lag_s,
    min_falloff_s=120,
    min_t_s_for_pick=90,
    max_analysis_s=180,
    pick_from="sqrt",      # "bourdet" | "sqrt" | "gfunc"

    # ---- NEW (optional) ----
    manual_shutins_surface=None,      # list of timestamps (str / pd.Timestamp)
    manual_windows_surface=None,      # list of (start_ts, end_ts)
    manual_tolerance_s=60,            # match window to cycle if within tol (for centers)
    prefer_manual=True,               # if True and a manual match exists -> use it
):
    """
    If manual_windows_surface is given: it overrides the [t_shut, t_end] per cycle (1:1 mapping by order).
    Else if manual_shutins_surface is given: each cycle's t_shut_in_surface is replaced by the nearest manual
    within manual_tolerance_s (in time); t_end is clipped to manual_shut + max_analysis_s.
    Otherwise: behaves exactly like before.
    """
    import numpy as np
    import pandas as pd

    def _to_ts(x):
        return pd.to_datetime(x) if not isinstance(x, pd.Timestamp) else x

    # normalize manuals
    manuals_centers = None
    manuals_windows = None
    if manual_shutins_surface:
        manuals_centers = [ _to_ts(x) for x in manual_shutins_surface ]
    if manual_windows_surface:
        manuals_windows = [ (_to_ts(a), _to_ts(b)) for (a, b) in manual_windows_surface ]

    tol = pd.to_timedelta(float(manual_tolerance_s), unit="s")
    results = []

    for k, c in enumerate(cycles, 1):
        # --- ORIGINAL cycle times
        t_shut_S = _to_ts(c['t_shut_in_surface'])
        t_end_S  = _to_ts(c['t_end_surface'])

        # --- OPTIONAL: override with manual *window* (strict 1:1 mapping by order)
        if prefer_manual and manuals_windows and k <= len(manuals_windows):
            t_shut_S, t_end_S = manuals_windows[k-1]

        # --- OPTIONAL: override with manual *center* matched by proximity
        elif prefer_manual and manuals_centers:
            # find nearest manual to this cycle's original shut-in
            diffs = [abs(mc - t_shut_S) for mc in manuals_centers]
            if len(diffs):
                j = int(np.argmin(diffs))
                if diffs[j] <= tol:
                    t_shut_S = manuals_centers[j]
                    # cap end to analysis window (do not exceed user cycle end)
                    t_end_S = min(t_end_S, t_shut_S + pd.to_timedelta(float(max_analysis_s), unit="s"))

        # --- Downhole timing
        t_shut_D = t_shut_S - pd.to_timedelta(float(lag_s), unit='s')
        t_end_D  = t_end_S  - pd.to_timedelta(float(lag_s), unit='s')

        # Build shut-in series from DH pressure
        ts_dh, p_dh = build_shut_in_series(time_D, p_downhole_corr, t_shut_D)

        # Hard cap the analysis window
        t_end_rel   = (pd.to_datetime(t_end_D) - pd.to_datetime(t_shut_D)).total_seconds()
        t_hard_cap  = min(float(t_end_rel), float(max_analysis_s))
        keep        = (ts_dh <= t_hard_cap)
        ts_dh, p_dh = ts_dh[keep], p_dh[keep]

        # Basic sanity
        if len(ts_dh) < 3 or (ts_dh[-1] if len(ts_dh) else 0) < min_falloff_s:
            results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                            'ended_by': c.get('ended_by', None), 'usable': False, 'reason': 'too_short'})
            continue

        # ---- PICKERS (unchanged) ----
        if pick_from == "bourdet":
            t_log, dP_dlogt = bourdet_derivative(ts_dh, p_dh, smooth_win=None, max_t_s=max_analysis_s)
            i_cl = suggest_closure_from_bourdet(t_log, dP_dlogt,
                                                min_t_s=min_t_s_for_pick, max_t_s=max_analysis_s)
            if i_cl is None:
                results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                                'ended_by': c.get('ended_by', None), 'usable': False,
                                'reason': 'no_closure_candidate_bourdet'})
                continue
            fcp    = float(p_dh[i_cl])
            t_cl_s = float(t_log[i_cl])

        elif pick_from == "sqrt":
            x_sqrt, p_srt, dpdx = derivative_vs_sqrt_time(ts_dh, p_dh, max_t_s=max_analysis_s)
            i_cl = suggest_closure_from_srt(x_sqrt, p_srt, dpdx,
                                            min_t_s=min_t_s_for_pick, max_t_s=max_analysis_s)
            if i_cl is None:
                results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                                'ended_by': c.get('ended_by', None), 'usable': False,
                                'reason': 'no_closure_candidate_sqrt'})
                continue
            fcp    = float(p_srt[i_cl])
            t_cl_s = float(x_sqrt[i_cl]**2)

        elif pick_from == "gfunc":
            tp_s = _tp_seconds_from_cycle(c)
            i_cl = suggest_closure_from_gfunction(
                ts_dh, p_dh, tp_s, min_t_s=min_t_s_for_pick, max_t_s=max_analysis_s
            )
            if i_cl is None:
                results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                                'ended_by': c.get('ended_by', None), 'usable': False,
                                'reason': 'no_closure_candidate_gfunc'})
                continue
            fcp    = float(p_dh[i_cl])
            t_cl_s = float(ts_dh[i_cl])

        else:
            results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                            'ended_by': c.get('ended_by', None), 'usable': False,
                            'reason': f'unknown_picker:{pick_from}'})
            continue

        results.append({
            'cycle': k,
            't_pump_start_surface': c.get('t_pump_start_surface', None),
            't_shut_in_surface': t_shut_S,
            't_end_surface': t_end_S,
            'ended_by': c.get('ended_by', None),
            'usable': True,
            'closure_time_s': t_cl_s,
            'closure_pressure_bar': fcp,
            'analysis_window_s': t_hard_cap,
            'picker': pick_from,
            'source': ('manual_window' if manuals_windows and k <= len(manuals_windows)
                       else ('manual_center' if manuals_centers and prefer_manual else 'auto'))
        })

    return pd.DataFrame(results)


# =========================================================
# Multi-cycle detection (pump-in → shut-in) with flow-back
# =========================================================

def _rate_from_cum(time_s: pd.Series, cum: pd.Series, out_units='m3/h'):
    """
    Finite-difference of a cumulative meter vs time (e.g., return volume),
    robust to counter resets and sampling spikes.
    """
    t = pd.to_datetime(time_s).astype('int64') / 1e9  # seconds
    c = pd.Series(cum, dtype=float)

    # keep only finite
    m = np.isfinite(t) & np.isfinite(c)
    t = t[m]; c = c[m]

    if len(t) < 2:
        return pd.Series(index=time_s.index, dtype=float)

    # MONOTONIZE to avoid negative jumps from counter resets
    c_mono = np.maximum.accumulate(c.to_numpy())

    dt = np.diff(t)                            # s
    dv = np.diff(c_mono)                       # m3 (non-negative)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = dv / np.where(dt == 0, np.nan, dt) # m3/s

    # light de-noising
    r = pd.Series(np.insert(r, 0, np.nan), index=time_s[m].index, dtype=float)
    r = r.rolling(3, center=True, min_periods=1).median()

    if out_units == 'm3/h':
        r = r * 3600.0

    # reindex back to full time base
    return r.reindex(time_s.index)


def detect_pump_cycles(
    time_S: pd.Series,
    q_m3h: pd.Series,
    return_vol_S: pd.Series | None = None,
    *,
    q_low=0.1, q_high=0.3,          # hysteresis thresholds (m3/h)
    min_hold_s=30,                  # must stay ≤ q_low for this long to confirm shut-in
    min_gap_s=60,                   # min idle gap before accepting next pump start
    flowback_rate_thresh=0.2,       # m3/h of returns growth ⇒ flow-back on

    # ---------- NEW (all optional) ----------
    manual_shutins_surface: list | None = None,     # list[str | pd.Timestamp] (centers)
    manual_windows_surface: list | None = None,     # list[(start_ts, end_ts)]
    manual_tolerance_s: float = 90.0,               # match center to nearest auto shut-in within tol
    prefer_manual: bool = True                      # if True, override autos with matched manuals
):
    """
    Detect all pump-in → shut-in cycles and cut each falloff at pump restart or
    flow-back start.

    Returns a list of dicts with:
      - t_pump_start_surface
      - t_shut_in_surface
      - t_end_surface
      - ended_by  ('restart' | 'flowback' | 'end' | 'manual')

    New behavior:
      * If `manual_windows_surface` is provided: return cycles defined by those windows
        (with pump-start inferred from the rate trace).
      * Else if `manual_shutins_surface` is provided: snap each manual to the sampled axis.
         - If within `manual_tolerance_s` of an auto shut-in and `prefer_manual=True`,
           override that cycle’s shut-in time.
         - Manuals with no matching auto create an extra cycle spanning from the last
           pump period before the manual to the next restart/flowback (or end of data).
    """
    import numpy as np
    import pandas as pd

    # ---- normalize inputs / clocks ----
    t = pd.to_datetime(pd.Series(time_S))
    q = pd.Series(q_m3h, dtype=float).reindex(t).astype(float)

    # Optional flow-back indicator from return volume
    rv_rate = None
    if return_vol_S is not None and pd.Series(return_vol_S).notna().any():
        rv_rate = _rate_from_cum(t, pd.Series(return_vol_S).reindex(t), out_units='m3/h')

    # ---------- helpers ----------
    def _is_flowback(i: int) -> bool:
        if rv_rate is None or not np.isfinite(q.iloc[i]):
            return False
        return (q.iloc[i] <= q_low) and (np.isfinite(rv_rate.iloc[i]) and rv_rate.iloc[i] > flowback_rate_thresh)

    def _snap_to_index(stamps) -> pd.DatetimeIndex:
        if not stamps:
            return pd.DatetimeIndex([])
        arr = t.values
        snapped = []
        for s in stamps:
            req = np.datetime64(pd.Timestamp(s))
            j = int(np.argmin(np.abs(arr - req)))
            snapped.append(pd.Timestamp(arr[j]))
        return pd.DatetimeIndex(snapped).unique().sort_values()

    def _nearest_pump_start(before_ts: pd.Timestamp | None) -> pd.Timestamp | None:
        """Walk backward to find a recent sustained pumping-on period (q >= q_high)."""
        if before_ts is None:
            return None
        i_end = int(np.searchsorted(t.values, before_ts.to_datetime64(), side='left')) - 1
        if i_end < 0: 
            return None
        # find last index where q >= q_high
        i = i_end
        while i >= 0 and not (np.isfinite(q.iloc[i]) and q.iloc[i] >= q_high):
            i -= 1
        if i < 0:
            return None
        # expand backward while still pumping
        j = i
        while j > 0 and np.isfinite(q.iloc[j-1]) and q.iloc[j-1] >= q_high:
            j -= 1
        # ensure a minimal gap from any previous cycle end by reusing min_gap_s logic is out-of-scope here
        return t.iloc[j]

    def _next_end_after(ts_start: pd.Timestamp) -> tuple[pd.Timestamp, str]:
        """Find next end due to flowback or restart; otherwise end-of-data."""
        i0 = int(np.searchsorted(t.values, ts_start.to_datetime64(), side='left'))
        for k in range(i0, len(t)):
            if _is_flowback(k):
                return t.iloc[k], 'flowback'
            if np.isfinite(q.iloc[k]) and q.iloc[k] >= q_high:
                return t.iloc[k], 'restart'
        return t.iloc[-1], 'end'

    tol = pd.to_timedelta(float(manual_tolerance_s), unit='s')
    manuals_centers = _snap_to_index(manual_shutins_surface) if manual_shutins_surface else pd.DatetimeIndex([])
    manuals_windows = [(pd.to_datetime(a), pd.to_datetime(b)) for (a, b) in manual_windows_surface] if manual_windows_surface else None

    # ---------- (A) If manual WINDOWS are provided: build cycles directly ----------
    if manuals_windows:
        out = []
        for (w_start, w_end) in manuals_windows:
            # infer pump start from rate just before the window (best-effort)
            t_pump_start = _nearest_pump_start(w_start)
            if t_pump_start is None:
                # fallback: the start of the window is the best we can do
                t_pump_start = w_start
            # sanity on end (don’t let it precede shut-in)
            w_end = max(w_end, w_start)
            out.append(dict(
                t_pump_start_surface=t_pump_start,
                t_shut_in_surface=w_start,
                t_end_surface=w_end,
                ended_by='manual'
            ))
        # drop trivial
        return [c for c in out if (c['t_end_surface'] - c['t_shut_in_surface']).total_seconds() >= min_hold_s]

    # ---------- (B) Otherwise: detect cycles automatically ----------
    cycles = []
    state = 'idle'
    t_pump_start = None
    last_end_t: pd.Timestamp | None = None
    t_shut = None  # keep latest confirmed shut-in

    for i in range(len(t)):
        ti = t.iloc[i]
        qi = q.iloc[i]
        if not np.isfinite(qi):
            continue

        # flow-back if pump ~0 AND returns increasing faster than threshold
        fb_on = _is_flowback(i)

        if state == 'idle':
            gap_ok = True if last_end_t is None else (ti - last_end_t).total_seconds() >= min_gap_s
            if qi >= q_high and gap_ok:
                t_pump_start = ti
                state = 'pumping'

        elif state == 'pumping':
            if qi <= q_low:
                # ensure it stays low for min_hold_s
                j, ok = i, True
                while j < len(t) and (t.iloc[j] - ti).total_seconds() <= min_hold_s:
                    if np.isfinite(q.iloc[j]) and q.iloc[j] > q_low:
                        ok = False; break
                    j += 1
                if ok:
                    t_shut = ti
                    state = 'shutin'

        if state == 'shutin':
            if fb_on:
                cycles.append(dict(
                    t_pump_start_surface=t_pump_start,
                    t_shut_in_surface=t_shut,
                    t_end_surface=ti,
                    ended_by='flowback'
                ))
                last_end_t = ti
                state = 'idle'
                t_pump_start = None
                t_shut = None
                continue

            if qi >= q_high:
                cycles.append(dict(
                    t_pump_start_surface=t_pump_start,
                    t_shut_in_surface=t_shut,
                    t_end_surface=ti,
                    ended_by='restart'
                ))
                last_end_t = ti
                state = 'pumping'
                t_pump_start = ti
                t_shut = None
                continue

    # close open falloff at end of data
    if state == 'shutin' and (t_shut is not None):
        end_t = t.iloc[-1]
        cycles.append(dict(
            t_pump_start_surface=t_pump_start,
            t_shut_in_surface=t_shut,
            t_end_surface=end_t,
            ended_by='end'
        ))

    # drop trivial cycles (too-short falloff)
    auto = []
    for c in cycles:
        if c['t_pump_start_surface'] is None:
            continue
        if (c['t_end_surface'] - c['t_shut_in_surface']).total_seconds() < min_hold_s:
            continue
        auto.append(c)

    # ---------- (C) Integrate MANUAL CENTERS (override or add) ----------
    if manuals_centers.empty:
        return auto

    # 1) override autos within tolerance
    if prefer_manual and len(auto) > 0:
        for c in auto:
            d = (manuals_centers - pd.to_datetime(c['t_shut_in_surface'])).abs()
            j = int(np.argmin(d)) if len(d) else None
            if j is not None and d[j] <= tol:
                # override shut-in timestamp (keep same end unless it precedes)
                c['t_shut_in_surface'] = manuals_centers[j]
                if c['t_end_surface'] < c['t_shut_in_surface']:
                    # re-find end after the manual time
                    c['t_end_surface'], c['ended_by'] = _next_end_after(c['t_shut_in_surface'])

    # 2) add manuals with no matching auto
    used = set()
    for c in auto:
        used.add(pd.to_datetime(c['t_shut_in_surface']))
    added = []
    for m in manuals_centers:
        # match by exact timestamp (after snapping) or by tolerance window to any overridden cycle
        is_matched = any(abs(m - pd.to_datetime(c['t_shut_in_surface'])) <= tol for c in auto)
        if is_matched:
            continue
        # create a new cycle around this manual shut-in
        ps = _nearest_pump_start(m)
        if ps is None:
            # cannot infer a reasonable pump start, skip this manual
            continue
        te, why = _next_end_after(m)
        if (te - m).total_seconds() < min_hold_s:
            continue
        added.append(dict(
            t_pump_start_surface=ps,
            t_shut_in_surface=m,
            t_end_surface=te,
            ended_by=why
        ))

    return sorted(auto + added, key=lambda c: c['t_shut_in_surface'])

def suggest_flow_thresholds(q_series: pd.Series):
    qpos = pd.Series(q_series, dtype=float)
    qpos = qpos[qpos > 0].dropna()
    if len(qpos) == 0:
        return 0.1, 0.3
    q_high = max(0.3, qpos.quantile(0.15))  # “definitely pumping”
    q_low  = max(0.1, qpos.quantile(0.02))  # “definitely off”
    return float(q_low), float(q_high)

def g_pick_by_linearity_break(
    G,
    semilog_dP,
    *,
    ts_seconds=None,
    min_t_s=0.0,
    max_t_s=None,
    early_frac=0.25,
    early_min=8,
    smooth_win=5,
    slope_window=7,
    slope_tol_rel=0.25,
    min_consec=3
):
    """
    Pick closure on the G-function by the *departure from linearity* of
    the semilog derivative (y = G * dP/dG) vs G, following Barree (2009).

    Parameters
    ----------
    G : array-like
        G-function values (monotonic increasing).
    semilog_dP : array-like
        Semilog derivative y = G * dP/dG sampled on G.
    ts_seconds : array-like or None
        Time since shut-in for each sample (optional; used for windowing).
    min_t_s, max_t_s : float or None
        Only consider points in [min_t_s, max_t_s] seconds if provided.
    early_frac : float
        Fraction of the valid leading samples to define the 'early linear' fit.
    early_min : int
        Minimum number of samples to include in the early linear fit.
    smooth_win : int (odd)
        Moving-average window (samples) to smooth semilog derivative for slope calc.
    slope_window : int (odd)
        Window length (samples) for local slope estimation by OLS.
    slope_tol_rel : float
        Relative tolerance for slope change: |m_local - m0| > slope_tol_rel*|m0|.
    min_consec : int
        Require this many consecutive samples breaching the slope tolerance.

    Returns
    -------
    dict with keys:
        ok : bool
        i_pick : int or None     # index in arrays (G, semilog_dP, ts_seconds)
        m0 : float               # early-time slope
        b0 : float               # early-time intercept
        m_local : np.ndarray     # local slope series (NaN where undefined)
        early_end_idx : int      # last index used in early fit
    """
    G = np.asarray(G, float)
    y = np.asarray(semilog_dP, float)

    n = len(G)
    if n < 5:
        return {"ok": False, "i_pick": None}

    finite = np.isfinite(G) & np.isfinite(y)
    if not finite.any():
        return {"ok": False, "i_pick": None}

    # Optional time windowing
    mask = finite.copy()
    if ts_seconds is not None:
        t = np.asarray(ts_seconds, float)
        mask &= np.isfinite(t)
        if min_t_s is not None:
            mask &= (t >= float(min_t_s))
        if (max_t_s is not None) and np.isfinite(max_t_s):
            mask &= (t <= float(max_t_s))

    idx = np.where(mask)[0]
    if len(idx) < 5:
        return {"ok": False, "i_pick": None}

    Gv = G[idx]
    yv = y[idx]

    # --- simple smoothing to reduce noise for slope estimation ---
    def _movavg(a, w):
        if w is None or w <= 1:
            return a.copy()
        w = int(w)
        if w % 2 == 0: w += 1
        k = w // 2
        out = np.full_like(a, np.nan, dtype=float)
        for i in range(len(a)):
            lo = max(0, i - k)
            hi = min(len(a), i + k + 1)
            seg = a[lo:hi]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                out[i] = seg.mean()
        return out

    yv_s = _movavg(yv, smooth_win)

    # --- define early linear segment ---
    early_n = max(int(np.ceil(early_frac * len(Gv))), int(early_min))
    early_n = min(early_n, len(Gv) - 2)  # leave room for break detection
    early_idx = np.arange(early_n)

    # OLS on early segment (exclude NaNs after smoothing)
    m = np.isfinite(yv_s[early_idx])
    if m.sum() < max(3, int(0.6*early_n)):
        # fallback to unsmoothed if smoothed is too sparse
        y_fit = yv[early_idx]
        mm = np.isfinite(y_fit)
        if mm.sum() < 3:
            return {"ok": False, "i_pick": None}
        p = np.polyfit(Gv[early_idx][mm], y_fit[mm], 1)
    else:
        p = np.polyfit(Gv[early_idx][m], yv_s[early_idx][m], 1)
    m0, b0 = float(p[0]), float(p[1])
    early_end_idx = idx[early_idx[-1]]

    # --- rolling local slope (OLS in a sliding window on smoothed series) ---
    w = int(slope_window)
    if w % 2 == 0: w += 1
    hw = w // 2
    m_local = np.full(len(Gv), np.nan, dtype=float)
    for i in range(len(Gv)):
        lo = max(0, i - hw)
        hi = min(len(Gv), i + hw + 1)
        xx = Gv[lo:hi]
        yy = yv_s[lo:hi]
        mm = np.isfinite(xx) & np.isfinite(yy)
        if mm.sum() >= max(3, w//2):
            pp = np.polyfit(xx[mm], yy[mm], 1)
            m_local[i] = pp[0]

    # tolerance on slope change
    slope_tol = float(slope_tol_rel) * max(abs(m0), 1e-12)

    # search for first persistent deviation after early window
    start = early_idx[-1] + 1
    if start >= len(Gv):
        return {"ok": False, "i_pick": None}

    def _first_persistent(breach):
        run = 0
        for j in range(start, len(Gv)):
            if breach[j]:
                run += 1
                if run >= int(min_consec):
                    return j - (min_consec - 1)
            else:
                run = 0
        return None

    breach = np.isfinite(m_local) & (np.abs(m_local - m0) > slope_tol)
    j_rel = _first_persistent(breach)
    if j_rel is None:
        return {"ok": False, "i_pick": None}

    # Map relative index back to absolute index in the original arrays
    i_pick = int(idx[j_rel])
    return {
        "ok": True,
        "i_pick": i_pick,
        "m0": m0,
        "b0": b0,
        "m_local": m_local,
        "early_end_idx": early_end_idx
    }


# --- add somewhere near other utilities ---
import numpy as np

def _finite_gradient(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if y.size < 2: 
        return np.full_like(y, np.nan, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    out[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dx0 = x[1]-x[0]; dx1 = x[-1]-x[-2]
    out[0]  = (y[1] - y[0]) / dx0 if dx0 != 0 else np.nan
    out[-1] = (y[-1] - y[-2]) / dx1 if dx1 != 0 else np.nan
    return out

import numpy as np
from typing import Dict, Any, Optional

def barree_tangent_pick(
    G: np.ndarray,
    P: np.ndarray,
    semilog_curve: Optional[np.ndarray] = None,
    smooth_window: int = 3,
    gmin_frac: float = 0.02,
    gmax_frac: float = 0.95,
    min_semilog: Optional[float] = None,   # <<< NEW: threshold on (flipped) semilog derivative, e.g. 15
) -> Dict[str, Any]:
    """
    Barree tangent (origin-line) pick on the (already sign-adjusted) semilog derivative curve.
    Assumes 'semilog_curve' is the *flipped positive* curve you actually plot (-G dP/dG).

    Returns:
      {"ok": bool, "m": slope, "Gc": Gc, "Pc": Pc, "idx": idx, "line": m*G, "debug": {...}}
    """

    dbg = {}
    out = {"ok": False, "debug": dbg}

    try:
        G = np.asarray(G, dtype=float)
        P = np.asarray(P, dtype=float)
        y = np.asarray(semilog_curve, dtype=float) if semilog_curve is not None else None

        if G.size < 3 or P.size != G.size or y is None or y.size != G.size:
            dbg["reason"] = "bad_inputs"
            return out

        # finite + sorted by G
        mfin = np.isfinite(G) & np.isfinite(P) & np.isfinite(y)
        G, P, y = G[mfin], P[mfin], y[mfin]
        if G.size < 3:
            dbg["reason"] = "too_few_points_after_finite"
            return out

        order = np.argsort(G)
        G, P, y = G[order], P[order], y[order]

        # basic window on G
        Gmin = np.nanquantile(G, np.clip(gmin_frac, 0.0, 0.49))
        Gmax = np.nanquantile(G, np.clip(gmax_frac, 0.51, 1.0))
        win = (G > 0) & (G >= Gmin) & (G <= Gmax)

        # optional smoothing (robust small median to tame noise)
        if smooth_window and smooth_window > 1 and smooth_window < G.size:
            k = int(smooth_window)
            k = k + 1 if (k % 2 == 0) else k   # median needs odd window
            # fast rolling median with pad via reflect
            pad = k // 2
            y_pad = np.pad(y, pad, mode="reflect")
            y_sm = np.empty_like(y)
            for i in range(y.size):
                y_sm[i] = np.median(y_pad[i:i+k])
            y = y_sm

        # NEW: threshold on semilog (already flipped positive upstream)
        if min_semilog is not None:
            win = win & (y >= float(min_semilog))

        # need enough points left
        if not np.any(win):
            dbg["reason"] = "no_points_after_threshold_window"
            dbg["Gmin"], dbg["Gmax"] = float(Gmin), float(Gmax)
            dbg["min_semilog"] = float(min_semilog) if min_semilog is not None else None
            return out

        # Barree origin-line tangent:
        # Find the *largest* slope m such that m*G <= y for all points in the window.
        # That m is min_i (y_i / G_i). Touch index is argmin(y/G).
        ratio = np.full_like(y, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio[win] = y[win] / G[win]

        # guard against pathological tiny G or negatives (already excluded by win)
        if not np.isfinite(ratio[win]).any():
            dbg["reason"] = "no_finite_ratio"
            return out

        idx = np.nanargmin(ratio)  # location where y/G is smallest (tightest tangent)
        m_star = float(ratio[idx])
        if not np.isfinite(m_star) or m_star <= 0.0:
            dbg["reason"] = "bad_slope"
            dbg["m_star"] = m_star
            return out

        Gc = float(G[idx])
        # interpolate Pc along P(G); use exact point if present
        Pc = float(np.interp(Gc, G, P)) if np.isfinite(Gc) else np.nan

        out.update({
            "ok": True,
            "m": m_star,
            "Gc": Gc,
            "Pc": Pc,
            "idx": int(idx),
            "y_semilog": y,
            "line": m_star * G,
            "debug": {
                "Gmin": float(Gmin), "Gmax": float(Gmax),
                "min_semilog": float(min_semilog) if min_semilog is not None else None,
                "n_used": int(np.count_nonzero(win))
            }
        })
        return out

    except Exception as e:
        dbg["reason"] = "exception"
        dbg["error"] = repr(e)
        return out

def haimson_bilinear_dpdt_vs_p(ts, P, min_left=8, min_right=8, smooth_window=3):
    """
    Haimson bilinear method on dP/dt vs P.
    Returns dict with ok, Pstar, slopeL, interceptL, slopeR, interceptR, split_idx, P_clean, dPdt_clean.
    """
    ts = np.asarray(ts, float); P = np.asarray(P, float)
    if ts.size < (min_left + min_right + 3) or P.size != ts.size:
        return {"ok": False}

    dPdt = _finite_gradient(P, ts)
    if (smooth_window is not None) and (smooth_window >= 3) and (smooth_window % 2 == 1) and (dPdt.size >= smooth_window):
        k = smooth_window // 2
        d_sm = dPdt.copy()
        for i in range(dPdt.size):
            d_sm[i] = np.nanmedian(dPdt[max(0, i-k):min(dPdt.size, i+k+1)])
        dPdt = d_sm

    msk = np.isfinite(P) & np.isfinite(dPdt)
    X, Y = P[msk], dPdt[msk]
    n = X.size
    if n < (min_left + min_right + 3):
        return {"ok": False}

    Sx, Sy = np.cumsum(X), np.cumsum(Y)
    Sxx, Sxy = np.cumsum(X*X), np.cumsum(X*Y)

    def fit_seg(i0, i1):
        m = i1 - i0 + 1
        if m <= 1:
            return (np.nan, np.nan, np.inf)
        sx  = Sx[i1]  - (Sx[i0-1]  if i0 > 0 else 0.0)
        sy  = Sy[i1]  - (Sy[i0-1]  if i0 > 0 else 0.0)
        sxx = Sxx[i1] - (Sxx[i0-1] if i0 > 0 else 0.0)
        sxy = Sxy[i1] - (Sxy[i0-1] if i0 > 0 else 0.0)
        denom = (m*sxx - sx*sx)
        if denom == 0:
            return (np.nan, np.nan, np.inf)
        a = (m*sxy - sx*sy) / denom
        b = (sy - a*sx) / m
        yhat = a*X[i0:i1+1] + b
        sse = float(np.sum((Y[i0:i1+1] - yhat)**2))
        return (a, b, sse)

    best = {"sse": np.inf}
    for split in range(min_left, n - min_right):
        aL, bL, sseL = fit_seg(0, split-1)
        aR, bR, sseR = fit_seg(split, n-1)
        if not (np.isfinite(aL) and np.isfinite(bL) and np.isfinite(aR) and np.isfinite(bR)):
            continue
        sse = sseL + sseR
        if sse < best["sse"]:
            best = {"sse": sse, "split": split, "aL": aL, "bL": bL, "aR": aR, "bR": bR}

    if not np.isfinite(best.get("sse", np.inf)):
        return {"ok": False}

    aL, bL = best["aL"], best["bL"]
    aR, bR = best["aR"], best["bR"]
    if (aL - aR) == 0:
        return {"ok": False}

    Pstar = (bR - bL) / (aL - aR)
    return {
        "ok": True,
        "Pstar": float(Pstar),
        "slopeL": float(aL), "interceptL": float(bL),
        "slopeR": float(aR), "interceptR": float(bR),
        "split_idx": int(best["split"]),
        "P_clean": X, "dPdt_clean": Y
    }