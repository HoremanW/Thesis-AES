import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# Shut-in detection (single)
# =========================
def find_shut_in_time_from_flow(time_surface, flow_m3h, threshold=0.1, min_hold_s=30):
    """
    Infer the shut-in timestamp from a surface flow signal (single event).
    See multi-cycle detector below for multiple shut-ins.
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


# ===========================================
# Build (t_s, p) relative to a given shut-in
#  - inserts an ISIP anchor at t = 0 s
#  - ensures strictly increasing time
# ===========================================
def build_shut_in_series(time, pressure, t_shut_in):
    """
    Construct (t_s, p) aligned to shut-in: seconds since shut-in vs. pressure.

    - Drops NaNs.
    - Trims to t >= 0.
    - Enforces strictly increasing time.
    - Inserts an anchor exactly at t=0 with an interpolated ISIP pressure
      if the first sample occurs after t=0.
    """
    t_series = pd.to_datetime(pd.Series(time))
    p_series = pd.Series(pressure, dtype=float)

    # Drop rows where either time or pressure is NaN
    m = t_series.notna() & p_series.notna()
    t_clean = t_series[m].reset_index(drop=True)
    p_clean = p_series[m].reset_index(drop=True)

    # Convert to seconds since shut-in
    t0 = pd.to_datetime(t_shut_in)
    ts = (t_clean - t0).dt.total_seconds().to_numpy()
    p  = p_clean.to_numpy()

    # Keep only times at/after shut-in
    keep = ts >= 0
    ts, p = ts[keep], p[keep]
    t_kept = t_clean[keep].reset_index(drop=True)

    # Enforce strictly increasing time (remove zero-time duplicates)
    if len(ts) >= 2:
        keep2 = np.concatenate(([True], np.diff(ts) > 0))
        ts, p = ts[keep2], p[keep2]
        t_kept = t_kept[keep2]

    # ---- Insert ISIP anchor at t=0 if first kept sample is after t=0 ----
    if len(ts) > 0 and ts[0] > 0:
        # find neighbors around t_shut_in in the *cleaned* series
        # left neighbor: last time <= t0; right neighbor: first time >= t0
        i_before = (t_clean <= t0).to_numpy().nonzero()[0]
        i_after  = (t_clean >= t0).to_numpy().nonzero()[0]

        if len(i_before) and len(i_after):
            i0, i1 = i_before[-1], i_after[0]
            t_left  = t_clean.iloc[i0].value
            t_right = t_clean.iloc[i1].value
            p_left  = p_clean.iloc[i0]
            p_right = p_clean.iloc[i1]
            if t_right != t_left and np.isfinite(p_left) and np.isfinite(p_right):
                w = (t0.value - t_left) / (t_right - t_left)
                p_at_shutin = (1 - w) * p_left + w * p_right
            else:
                p_at_shutin = p_clean.iloc[i0]
        else:
            # fallback: use first kept sample value
            p_at_shutin = p[0]

        ts = np.insert(ts, 0, 0.0)
        p  = np.insert(p,  0, float(p_at_shutin))

    return ts, p


# ===========================================
# Bourdet derivative
# ===========================================
def bourdet_derivative(t_seconds, p, smooth_win=None):
    """
    Compute Bourdet derivative dP/d(ln t) for irregularly sampled pressure drawdown.
    """
    t = np.asarray(t_seconds, float)
    pp = np.asarray(p, float)

    if smooth_win and smooth_win >= 3 and (smooth_win % 2 == 1):
        from scipy.signal import medfilt
        pp = medfilt(pp, kernel_size=smooth_win)

    if len(t) < 3:
        return t, np.full_like(t, np.nan)

    ln_t = np.log(t, where=t > 0, out=np.full_like(t, np.nan))

    dp = np.empty_like(pp)
    dp[:] = np.nan

    # Central Bourdet derivative (uses neighbors i-1 and i+1 in ln-space)
    for i in range(1, len(t) - 1):
        i0, i2 = i - 1, i + 1
        denom = ln_t[i2] - ln_t[i0]
        if np.isfinite(denom) and denom != 0:
            left = (pp[i] - pp[i0]) / (ln_t[i] - ln_t[i0])
            right = (pp[i2] - pp[i]) / (ln_t[i2] - ln_t[i])
            dp[i] = 0.5 * (left + right)

    return t, dp


# ===========================================
# √t diagnostic
# ===========================================
def derivative_vs_sqrt_time(t_seconds, p):
    """
    Compute pressure derivative with respect to sqrt(time): dP/d(√t).
    """
    t = np.asarray(t_seconds, float)
    pp = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(pp) & (t >= 0)
    t, pp = t[m], pp[m]

    if len(t) < 3:
        return np.sqrt(t), pp, np.full_like(t, np.nan)

    x = np.sqrt(t)

    dpdx = np.empty_like(pp)
    dpdx[:] = np.nan

    # Central differences in √t
    for i in range(1, len(x) - 1):
        dx = x[i + 1] - x[i - 1]
        if dx != 0:
            dpdx[i] = (pp[i + 1] - pp[i - 1]) / dx

    return x, pp, dpdx


def suggest_closure_from_srt(
    x_sqrt_t,
    p,
    dpdx,
    *,
    min_t_s=1,
    max_t_s=None,          # set to 180 when you want to enforce 3-minute cap here too
    guard_n=2,             # samples to keep away from edges when checking local shape
    prefer_global_min=True,
    dpdx_smooth_pts=5      # rolling median points; set to None/1 to disable
):
    """
    Pick a closure index from √t diagnostics by taking the (smoothed) GLOBAL
    MINIMUM of dP/d√t within [min_t_s, max_t_s], with sensible guards.

    Returns
    -------
    int or None
    """
    import numpy as np
    import pandas as pd

    x = np.asarray(x_sqrt_t, float)
    d = np.asarray(dpdx, float)

    # Time window in seconds since shut-in
    t = x**2
    win = np.isfinite(d) & (t >= float(min_t_s))
    if max_t_s is not None:
        win &= (t <= float(max_t_s))

    if not np.any(win):
        return None

    # Optional robust smoothing of the derivative to avoid single-sample spikes
    if dpdx_smooth_pts and dpdx_smooth_pts >= 3:
        d_s = pd.Series(d).rolling(window=int(dpdx_smooth_pts), center=True, min_periods=1).median().to_numpy()
    else:
        d_s = d

    idx = np.where(win)[0]

    if prefer_global_min:
        # Global argmin inside window
        i_glob = idx[np.nanargmin(d_s[idx])]

        # If the global min is usable (not on the edge and shows an upturn), take it
        if (i_glob - idx[0] >= guard_n) and (idx[-1] - i_glob >= guard_n):
            if np.isfinite(d_s[i_glob-1]) and np.isfinite(d_s[i_glob+1]):
                if d_s[i_glob-1] > d_s[i_glob] < d_s[i_glob+1]:
                    return int(i_glob)

        # Otherwise, find the best local minimum within the window
        cand = None
        best = np.inf
        for j in idx[guard_n: len(idx)-guard_n]:
            if np.isfinite(d_s[j]) and (d_s[j-1] > d_s[j] < d_s[j+1]):
                if d_s[j] < best:
                    best = d_s[j]
                    cand = j
        if cand is not None:
            return int(cand)

        # Fallback: return global min even if on edge (better than None)
        return int(i_glob)

    # Legacy behavior: first pronounced local minimum after min_t_s
    for j in idx[guard_n: len(idx)-guard_n]:
        if np.isfinite(d_s[j]) and (d_s[j-1] > d_s[j] < d_s[j+1]):
            return int(j)

    return None



# ===========================================
# Barree/Nolte high-efficiency G-function
# ===========================================
def g_function_high_efficiency(delta_t_seconds: np.ndarray, tp_seconds: float) -> np.ndarray:
    """
    High-efficiency (low-leakoff) G-function (Appendix A).
    """
    d = np.asarray(delta_t_seconds, float) / max(1e-12, float(tp_seconds))
    root = np.sqrt(1.0 + d)
    G = (4.0 / 3.0) * (root - 1.0) ** 2 + (root - 1.0)
    return G


def semilog_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute “semilog derivative” X * dY/dX (Barree plotting convention).
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


# =========================================================
# Multi-cycle detection (pump-in → shut-in) with flow-back
# =========================================================
def _rate_from_cum(time_s: pd.Series, cum: pd.Series, out_units='m3/h'):
    """Finite-difference derivative of a cumulative counter versus time."""
    t = pd.to_datetime(time_s).astype('int64') / 1e9  # seconds
    c = pd.Series(cum, dtype=float)
    m = np.isfinite(t) & np.isfinite(c)
    t, c = t[m], c[m]
    if len(t) < 2:
        return pd.Series(index=time_s.index, dtype=float)
    dt = np.diff(t)  # seconds
    dv = np.diff(c)
    r = np.empty_like(c, dtype=float)
    r[:] = np.nan
    r[1:] = dv / np.where(dt == 0, np.nan, dt)  # m3/s if cum in m3
    if out_units == 'm3/h':
        r = r * 3600.0
    out = pd.Series(r, index=time_s[m].index, dtype=float)
    return out.reindex(time_s.index)  # align back to original index


def detect_pump_cycles(
    time_S: pd.Series,
    q_m3h: pd.Series,
    return_vol_S: pd.Series | None = None,
    *,
    q_low=0.1, q_high=0.3,        # hysteresis (m3/h)
    min_hold_s=30,
    min_gap_s=60,
    flowback_rate_thresh=0.2      # m3/h of returns growth ⇒ flow-back on
):
    """
    Find all pump-in → shut-in cycles and their 'clean falloff' windows.

    Returns a list of dicts with:
      - t_pump_start_surface
      - t_shut_in_surface
      - t_end_surface          (first of pump restart or flow-back start)
      - ended_by               ('restart', 'flowback', or 'end')
    """
    t = pd.to_datetime(pd.Series(time_S))
    q = pd.Series(q_m3h, dtype=float)

    # (optional) estimate return line rate to detect flow-back
    rv_rate = None
    if return_vol_S is not None and return_vol_S.notna().any():
        rv_rate = _rate_from_cum(t, return_vol_S, out_units='m3/h')

    cycles = []
    state = 'idle'
    t_pump_start = None

    # IMPORTANT: avoid OutOfBoundsDatetime by using None until first end
    last_end_t: pd.Timestamp | None = None

    for i in range(len(t)):
        ti = t.iloc[i]
        qi = q.iloc[i]
        if not np.isfinite(qi):
            continue

        # detect flowback (returns increasing while pump is ~off)
        fb_on = False
        if rv_rate is not None and np.isfinite(rv_rate.iloc[i]):
            fb_on = (qi <= q_low) and (rv_rate.iloc[i] > flowback_rate_thresh)

        if state == 'idle':
            # Start pumping when we cross above q_high.
            # If we already ended a previous cycle, also enforce min_gap_s.
            gap_ok = True if last_end_t is None else (ti - last_end_t).total_seconds() >= min_gap_s
            if qi >= q_high and gap_ok:
                t_pump_start = ti
                state = 'pumping'

        elif state == 'pumping':
            # candidate shut-in when we drop below q_low and stay low for min_hold_s
            if qi <= q_low:
                j = i
                ok = True
                while j < len(t) and (t.iloc[j] - ti).total_seconds() <= min_hold_s:
                    if np.isfinite(q.iloc[j]) and q.iloc[j] > q_low:
                        ok = False
                        break
                    j += 1
                if ok:
                    t_shut = ti
                    state = 'shutin'

        if state == 'shutin':
            # end conditions: (1) pump restarts, or (2) flow-back starts
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
                state = 'pumping'  # a new stage may be starting immediately
                t_pump_start = ti
                continue

    # flush if still in shut-in at end of data
    if state == 'shutin':
        end_t = t.iloc[-1]
        cycles.append(dict(
            t_pump_start_surface=t_pump_start,
            t_shut_in_surface=t_shut,
            t_end_surface=end_t,
            ended_by='end'
        ))
        last_end_t = end_t

    # sanity filter: drop trivial cycles (no pumping or no falloff)
    out = []
    for c in cycles:
        if c['t_pump_start_surface'] is None:
            continue
        if (c['t_end_surface'] - c['t_shut_in_surface']).total_seconds() < min_hold_s:
            continue
        out.append(c)

    return out



# =========================================================
# Analyze all shut-ins → closure pressure per cycle
#  - trims each falloff to (pump restart | flow-back start | end)
#  - uses your SRT/Bourdet heuristic
# =========================================================
def analyze_all_shutins(
    cycles,
    *,
    time_S, flowrate_S, return_volume_S,
    time_D, p_downhole_corr,
    lag_s,
    min_falloff_s=120,         # require ≥ 2 min of clean falloff
    min_t_s_for_pick=90,       # ignore earliest 90 s for closure heuristic
    max_analysis_s=180,        # <-- NEW: hard cap analysis to first 3 minutes
):
    results = []

    for k, c in enumerate(cycles, 1):
        t_shut_S = c['t_shut_in_surface']
        t_end_S  = c['t_end_surface']

        # convert window to DOWNHOLE clock
        t_shut_D = t_shut_S - pd.to_timedelta(float(lag_s), unit='s')
        t_end_D  = t_end_S  - pd.to_timedelta(float(lag_s), unit='s')

        # build DH falloff
        ts_dh, p_dh = build_shut_in_series(time_D, p_downhole_corr, t_shut_D)

        # trim to (cycle end) AND to max_analysis_s after shut-in (first 3 minutes)
        t_end_rel = (pd.to_datetime(t_end_D) - pd.to_datetime(t_shut_D)).total_seconds()
        t_hard_cap = min(float(t_end_rel), float(max_analysis_s))
        keep = (ts_dh <= t_hard_cap)
        ts_dh, p_dh = ts_dh[keep], p_dh[keep]

        # require minimum clean falloff (still makes sense with 180 s cap)
        if len(ts_dh) < 3 or (ts_dh[-1] if len(ts_dh) else 0) < min_falloff_s:
            results.append({
                'cycle': k,
                't_shut_in_surface': t_shut_S,
                'ended_by': c['ended_by'],
                'usable': False,
                'reason': 'too_short'
            })
            continue

        # √t diagnostic and Bourdet
        x_sqrt, p_srt, dpdx = derivative_vs_sqrt_time(ts_dh, p_dh)
        i_cl = suggest_closure_from_srt(
            x_sqrt, p_srt, dpdx,
            min_t_s=min_t_s_for_pick,
            max_t_s=max_analysis_s,   # respect your 3-minute cap inside the picker too
            guard_n=5,
            prefer_global_min=True,
            dpdx_smooth_pts=5
        )


        if i_cl is None:
            results.append({
                'cycle': k,
                't_shut_in_surface': t_shut_S,
                'ended_by': c['ended_by'],
                'usable': False,
                'reason': 'no_closure_candidate'
            })
            continue

        fcp = float(p_srt[i_cl])
        t_cl_s = float(x_sqrt[i_cl] ** 2)

        results.append({
            'cycle': k,
            't_pump_start_surface': c['t_pump_start_surface'],
            't_shut_in_surface': t_shut_S,
            't_end_surface': t_end_S,
            'ended_by': c['ended_by'],
            'usable': True,
            'closure_time_s': t_cl_s,
            'closure_pressure_bar': fcp,
            'analysis_window_s': t_hard_cap,   # for transparency/debug
        })

    return pd.DataFrame(results)

