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


def build_shut_in_series(time, pressure, t_shut_in, max_seconds=None):
    """
    Construct (t_s, p) aligned to shut-in: seconds since shut-in vs. pressure.

    This cleans and synchronizes the time/pressure arrays, drops NaNs, computes
    time in seconds relative to `t_shut_in`, trims negative times, and enforces
    strictly increasing time (removing any zero/duplicate time steps).

    Parameters
    ----------
    time : array-like (datetime64-like or strings convertible by pandas)
        Absolute timestamps for downhole/surface pressure measurements.
    pressure : array-like of float
        Pressure values (e.g., bar). NaNs are dropped alongside their times.
    t_shut_in : datetime-like
        The reference shut-in time (pandas/NumPy/pydatetime accepted).

    Returns
    -------
    t_s : np.ndarray
        Seconds since shut-in (>= 0), strictly increasing.
    p : np.ndarray
        Pressure values corresponding to `t_s`.

    Notes
    -----
    - If `t_shut_in` is not within the range of `time`, you may get few or no
      samples after trimming.
    - If multiple readings share identical timestamps after rounding, only the
      first of each identical pair is kept to ensure monotonic time.
    """
    t = pd.to_datetime(pd.Series(time))
    p = pd.Series(pressure, dtype=float)

    # Drop rows where either time or pressure is NaN
    m = t.notna() & p.notna()
    t, p = t[m].reset_index(drop=True), p[m].reset_index(drop=True)

    # Convert to seconds since shut-in
    ts = (t - pd.to_datetime(t_shut_in)).dt.total_seconds().to_numpy()

    # Keep only times at/after shut-in
    keep = ts >= 0
    ts, p = ts[keep], p.to_numpy()[keep]

    # Enforce strictly increasing time (remove zero-time duplicates)
    if len(ts) >= 2:
        keep2 = np.concatenate(([True], np.diff(ts) > 0))
        ts, p = ts[keep2], p[keep2]
    
    if max_seconds is not None:
        keep3 = ts <= float(max_seconds)
        ts, p = ts[keep3], p[keep3]


    return ts, p


def bourdet_derivative(t_seconds, p, smooth_win=None, max_t_s=None):
    """
    Compute Bourdet derivative dP/d(ln t) for irregularly sampled pressure drawdown.

    The Bourdet derivative is a central-difference approximation on a log-time
    axis, commonly used in pressure transient analysis for better stability on
    irregular sampling.

    Parameters
    ----------
    t_seconds : array-like of float
        Time since shut-in in seconds (t > 0 recommended).
    p : array-like of float
        Pressure values aligned to `t_seconds`.
    smooth_win : int or None, optional
        If provided and is an odd integer >= 3, apply a median filter of this
        window length to `p` before differentiation (requires SciPy).

    Returns
    -------
    t_mid : np.ndarray
        Same length as input `t_seconds`. (Central points are meaningful;
        endpoints are diffed forward/backward and may be less accurate.)
    dp_dlogt : np.ndarray
        Bourdet derivative dP/d(ln t); NaN where not computable.

    Notes
    -----
    - Requires at least 3 valid points to compute central differences.
    - `t` values <= 0 are invalid on log-scale and yield NaNs.
    - Consider light smoothing (`smooth_win=3 or 5`) for noisy data.
    """
    t  = np.asarray(t_seconds, float)
    pp = np.asarray(p,          float)

    # Hard cap window
    if max_t_s is not None:
        mcap = np.isfinite(t) & (t <= float(max_t_s))
        t, pp = t[mcap], pp[mcap]

    if smooth_win and smooth_win >= 3 and (smooth_win % 2 == 1):
        try:
            from scipy.signal import medfilt
            pp = medfilt(pp, kernel_size=int(smooth_win))
        except Exception:
            pass

    if len(t) < 3:
        return t, np.full_like(t, np.nan)

    ln_t = np.log(t, where=t > 0, out=np.full_like(t, np.nan))

    dp = np.empty_like(pp)
    dp[:] = np.nan
    for i in range(1, len(t) - 1):
        i0, i2 = i - 1, i + 1
        denom = ln_t[i2] - ln_t[i0]
        if np.isfinite(denom) and denom != 0:
            left  = (pp[i]  - pp[i0]) / (ln_t[i]  - ln_t[i0])
            right = (pp[i2] - pp[i])  / (ln_t[i2] - ln_t[i])
            dp[i] = 0.5 * (left + right)
    return t, dp

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
    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    if max_t_s is not None:
        m &= (t <= float(max_t_s))
    t, pp = t[m], pp[m]

    if len(t) < 3:
        return np.sqrt(t), pp, np.full_like(t, np.nan)

    sqrt_t = np.sqrt(t)
    x = sqrt_t

    dpdx = np.empty_like(pp)
    dpdx[:] = np.nan

    # Central differences in √t
    for i in range(1, len(x) - 1):
        dx = x[i + 1] - x[i - 1]
        if dx != 0:
            dpdx[i] = (pp[i + 1] - pp[i - 1]) / dx

    return x, pp, dpdx

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
    """Finite-difference of a cumulative meter vs time (e.g., return volume)."""
    t = pd.to_datetime(time_s).astype('int64') / 1e9  # seconds
    c = pd.Series(cum, dtype=float)
    m = np.isfinite(t) & np.isfinite(c)
    t, c = t[m], c[m]
    if len(t) < 2:
        return pd.Series(index=time_s.index, dtype=float)
    dt = np.diff(t)                        # s
    dv = np.diff(c)                        # m3
    r = np.full_like(c, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r[1:] = dv / dt                    # m3/s
    if out_units == 'm3/h':
        r *= 3600.0
    out = pd.Series(r, index=time_s[m].index, dtype=float)
    return out.reindex(time_s.index)       # align back to original index


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

