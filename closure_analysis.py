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


def fcp_by_sqrt_intersection(t_seconds, p, *, max_t_s=180, min_left=8, min_right=8):
    """
    Closure by intersection on P vs √t.
    Returns dict with closure pressure and time if ok, else ok=False.
    """
    t = np.asarray(t_seconds, float)
    pp = np.asarray(p, float)

    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    t, pp = t[m], pp[m]
    if len(t) < min_left + min_right + 1:
        return {'ok': False, 'reason': 'too_few_points'}

    x = np.sqrt(t)
    fit = _two_segment_intersection(x, pp, min_left=min_left, min_right=min_right)
    if not fit.get('ok', False):
        return fit

    # convert x* back to time (seconds)
    t_star = max(0.0, fit['x_star']**2)
    return {
        'ok': True,
        'closure_time_s': float(t_star),
        'closure_pressure_bar': float(fit['y_star']),
        'break_index': fit['i_break'],
        'fit': fit
    }

def fcp_by_g_intersection(t_seconds, p, tp_seconds, *, max_t_s=180, min_left=8, min_right=8):
    """
    Closure by intersection on P vs G(t/tp).
    Returns dict with closure pressure and time if ok, else ok=False.
    """
    t = np.asarray(t_seconds, float)
    pp = np.asarray(p, float)

    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    t, pp = t[m], pp[m]
    if len(t) < min_left + min_right + 1:
        return {'ok': False, 'reason': 'too_few_points'}

    # Build G on the same t grid
    G = g_function_high_efficiency(t, tp_seconds=max(1.0, float(tp_seconds)))
    fit = _two_segment_intersection(G, pp, min_left=min_left, min_right=min_right)
    if not fit.get('ok', False):
        return fit

    # Convert x*=G* back to time is non-trivial analytically; we can map by nearest index
    # (closure pressure is what matters; time is approximate)
    # Nearest t where G≈G*:
    i_near = int(np.nanargmin(np.abs(G - fit['x_star'])))
    t_star = float(t[i_near])

    return {
        'ok': True,
        'closure_time_s': t_star,
        'closure_pressure_bar': float(fit['y_star']),
        'break_index': fit['i_break'],
        'fit': fit
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
    pick_from="bourdet",      # "bourdet" | "sqrt" | "gfunc"
):
    results = []
    for k, c in enumerate(cycles, 1):
        t_shut_S = c['t_shut_in_surface']
        t_end_S  = c['t_end_surface']

        t_shut_D = t_shut_S - pd.to_timedelta(float(lag_s), unit='s')
        t_end_D  = t_end_S  - pd.to_timedelta(float(lag_s), unit='s')

        ts_dh, p_dh = build_shut_in_series(time_D, p_downhole_corr, t_shut_D)

        t_end_rel   = (pd.to_datetime(t_end_D) - pd.to_datetime(t_shut_D)).total_seconds()
        t_hard_cap  = min(float(t_end_rel), float(max_analysis_s))
        keep        = (ts_dh <= t_hard_cap)
        ts_dh, p_dh = ts_dh[keep], p_dh[keep]

        if len(ts_dh) < 3 or (ts_dh[-1] if len(ts_dh) else 0) < min_falloff_s:
            results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                            'ended_by': c['ended_by'], 'usable': False, 'reason': 'too_short'})
            continue

        if pick_from == "bourdet":
            t_log, dP_dlogt = bourdet_derivative(ts_dh, p_dh, smooth_win=None, max_t_s=max_analysis_s)
            i_cl = suggest_closure_from_bourdet(t_log, dP_dlogt,
                                                min_t_s=min_t_s_for_pick, max_t_s=max_analysis_s)
            if i_cl is None:
                results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                                'ended_by': c['ended_by'], 'usable': False,
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
                                'ended_by': c['ended_by'], 'usable': False,
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
                                'ended_by': c['ended_by'], 'usable': False,
                                'reason': 'no_closure_candidate_gfunc'})
                continue
            fcp    = float(p_dh[i_cl])
            t_cl_s = float(ts_dh[i_cl])

        else:
            results.append({'cycle': k, 't_shut_in_surface': t_shut_S,
                            'ended_by': c['ended_by'], 'usable': False,
                            'reason': f'unknown_picker:{pick_from}'})
            continue

        results.append({
            'cycle': k,
            't_pump_start_surface': c['t_pump_start_surface'],
            't_shut_in_surface': t_shut_S,
            't_end_surface': t_end_S,
            'ended_by': c['ended_by'],
            'usable': True,
            'closure_time_s': t_cl_s,
            'closure_pressure_bar': fcp,
            'analysis_window_s': t_hard_cap,
            'picker': pick_from,
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
    q_low=0.1, q_high=0.3,    # hysteresis thresholds (m3/h)
    min_hold_s=30,            # must stay ≤ q_low for this long to confirm shut-in
    min_gap_s=60,             # min idle gap before accepting next pump start
    flowback_rate_thresh=0.2  # m3/h of returns growth ⇒ flow-back on
):
    """
    Detect all pump-in → shut-in cycles and cut each falloff at pump restart or
    flow-back start. Returns a list of dicts with:
      - t_pump_start_surface
      - t_shut_in_surface
      - t_end_surface
      - ended_by  ('restart' | 'flowback' | 'end')
    """
    t = pd.to_datetime(pd.Series(time_S))
    q = pd.Series(q_m3h, dtype=float)

    # Optional flow-back indicator from return volume
    rv_rate = None
    if return_vol_S is not None and return_vol_S.notna().any():
        rv_rate = _rate_from_cum(t, return_vol_S, out_units='m3/h')

    cycles = []
    state = 'idle'
    t_pump_start = None
    last_end_t: pd.Timestamp | None = None

    for i in range(len(t)):
        ti = t.iloc[i]
        qi = q.iloc[i]
        if not np.isfinite(qi):
            continue

        # flow-back if pump ~0 AND returns increasing faster than threshold
        fb_on = False
        if rv_rate is not None and np.isfinite(rv_rate.iloc[i]):
            fb_on = (qi <= q_low) and (rv_rate.iloc[i] > flowback_rate_thresh)

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
                continue

    # close open falloff at end of data
    if state == 'shutin':
        end_t = t.iloc[-1]
        cycles.append(dict(
            t_pump_start_surface=t_pump_start,
            t_shut_in_surface=t_shut,
            t_end_surface=end_t,
            ended_by='end'
        ))

    # drop trivial cycles (too-short falloff)
    out = []
    for c in cycles:
        if c['t_pump_start_surface'] is None:
            continue
        if (c['t_end_surface'] - c['t_shut_in_surface']).total_seconds() < min_hold_s:
            continue
        out.append(c)
    return out

def suggest_flow_thresholds(q_series: pd.Series):
    qpos = pd.Series(q_series, dtype=float)
    qpos = qpos[qpos > 0].dropna()
    if len(qpos) == 0:
        return 0.1, 0.3
    q_high = max(0.3, qpos.quantile(0.15))  # “definitely pumping”
    q_low  = max(0.1, qpos.quantile(0.02))  # “definitely off”
    return float(q_low), float(q_high)