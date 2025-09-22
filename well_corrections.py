import numpy as np
import pandas as pd

g = 9.80665  # gravitational acceleration [m/s^2]

def series_to_seconds(t):
    """
    Convert a datetime-like array/Series to seconds since the first timestamp.
    """
    t = pd.to_datetime(pd.Series(t), errors='coerce')
    if t.isna().all():
        raise ValueError("series_to_seconds: all timestamps are NaT after coercion.")
    t = t.ffill()  # if any internal NaT, forward-fill minimally
    return (t - t.iloc[0]).dt.total_seconds().to_numpy()


def align_series(src_time, src_vals, dst_time, kind='linear'):
    """
    Interpolate (or step) values defined on `src_time` onto `dst_time`.
    """
    ts_src = series_to_seconds(src_time)
    ts_dst = series_to_seconds(dst_time)

    # Drop non-increasing stamps
    keep = np.concatenate(([True], np.diff(ts_src) > 0))
    ts_src = ts_src[keep]
    src_vals = pd.Series(src_vals).to_numpy()[keep]

    # Fallbacks for tiny input
    if ts_src.size == 0:
        raise ValueError("align_series: no valid source timestamps after cleaning.")
    if ts_src.size == 1:
        y = np.full_like(ts_dst, fill_value=src_vals[0], dtype=float)
    else:
        y = np.interp(ts_dst, ts_src, src_vals, left=src_vals[0], right=src_vals[-1])

    return pd.Series(y, index=pd.Series(dst_time).index)


def hydrostatic_correct_to_fracture(
    p_gauge, time_gauge,
    rho_surface, time_surface,
    delta_tvd_m,
    out_units='bar',
    lag_s=None  # if downhole (gauge) lags surface by lag_s (>0), shift gauge times BACK by lag_s
):
    """
    Hydrostatic correction: P_fracture(t) = P_gauge(t) + ρ(t) * g * ΔTVD.
    """

    rho_surface.loc[:] = rho_surface.iloc[1]
    # Optionally shift gauge times to align with surface density sampling
    tg = pd.to_datetime(pd.Series(time_gauge), errors='coerce')
    if lag_s is not None:
        tg = tg - pd.to_timedelta(lag_s, unit='s')

    # Density cleanup and unit heuristic
    rho = pd.Series(rho_surface).astype(float)
    if rho.dropna().median() < 20:
        rho = rho * 1000.0  # g/cm^3 -> kg/m^3

    # Interpolate density to the (possibly shifted) gauge times
    rho_on_gauge = align_series(time_surface, rho, tg).to_numpy()

    # Hydrostatic increment [Pa]
    dP_Pa = rho_on_gauge * g * float(delta_tvd_m)

    out = (out_units or '').lower()
    if out == 'bar':
        dP = dP_Pa / 1e5
    elif out == 'mpa':
        dP = dP_Pa / 1e6
    else:
        dP = dP_Pa  # Pa

    p_gauge = pd.Series(p_gauge).astype(float).to_numpy()
    p_at_fracture = p_gauge + dP

    # Return aligned with the ORIGINAL (unshifted) gauge index
    idx = pd.Series(time_gauge).index
    return (
        pd.Series(p_at_fracture, index=idx),
        pd.Series(dP,           index=idx),
    )


def darcy_weisbach_friction_drop(
    rho,               # kg/m^3
    Q,                 # volumetric flow rate [m^3/s]
    L,                 # length [m]
    path='tubing',     # 'tubing' or 'annulus'
    D=None,            # pipe ID [m] if path='tubing'
    Do=None, Di=None,  # annulus diameters [m]
    mu=None,           # viscosity [Pa·s] (if f not given)
    roughness=0.0,     # absolute roughness ε [m]
    f=None             # Darcy friction factor (if provided, skips Re/roughness calc)
):
    """
    ΔP_fric = f · (L/D_h) · (ρ v² / 2)
    """
    # Geometry
    if path.lower() == 'annulus':
        if Do is None or Di is None:
            raise ValueError("For annulus, provide Do and Di (meters).")
        if Do <= Di:
            raise ValueError("Annulus requires Do > Di.")
        A = 0.25 * np.pi * (Do**2 - Di**2)
        Dh = Do - Di
    else:
        if D is None:
            raise ValueError("For tubing, provide D (ID in meters).")
        A = 0.25 * np.pi * D**2
        Dh = D

    # Velocity
    v = Q / A

    # If no flow, no friction drop
    if np.isclose(v, 0.0):
        return 0.0, {'A': A, 'Dh': Dh, 'v': v, 'Re': 0.0, 'f': 0.0}

    # Friction factor
    Re = None
    if f is None:
        if mu is None or mu <= 0:
            raise ValueError("Provide mu (Pa·s) or directly provide f (Darcy).")
        Re = abs(rho * v * Dh / mu)
        if Re < 2300:
            f = 64.0 / max(Re, 1e-12)  # avoid div/0
        else:
            eps = max(float(roughness), 0.0)
            f = 0.25 / (np.log10((eps / (3.7 * Dh)) + (5.74 / (Re**0.9))))**2

    dP_pa = f * (L / Dh) * 0.5 * rho * v**2
    return float(dP_pa), {'A': A, 'Dh': Dh, 'v': v, 'Re': Re, 'f': f}


def estimate_lag(MD, TVD, gauge_index, TVD_fracture_m):
    """
    Estimate the lag (ΔTVD) between a gauge and a fracture depth.

    Parameters
    ----------
    MD : array-like
        Measured Depth values (same length as TVD).
    TVD : array-like
        True Vertical Depth values; can contain NaN for missing values.
    gauge_index : int
        Index of the gauge depth in the MD/TVD arrays.
    TVD_fracture_m : float
        Known TVD of the fracture (m).

    Returns
    -------
    TVD_interp : np.ndarray
        TVD values with NaNs linearly filled along MD.
    TVD_gauge_m : float
        Interpolated TVD at the gauge MD.
    delta_tvd_m : float
        TVD_fracture_m - TVD_gauge_m.
    """
    MD = np.asarray(MD, dtype=float)
    TVD = np.asarray(TVD, dtype=float)

    if np.isnan(TVD).all():
        raise ValueError("estimate_lag: all TVD values are NaN.")
    if gauge_index < 0 or gauge_index >= MD.size:
        raise IndexError("estimate_lag: gauge_index out of range.")

    TVD_interp = np.interp(MD, MD[~np.isnan(TVD)], TVD[~np.isnan(TVD)])
    TVD_gauge_m = float(TVD_interp[gauge_index])
    delta_tvd_m = float(TVD_fracture_m) - TVD_gauge_m

    return TVD_interp, TVD_gauge_m, delta_tvd_m

# ------------- NEW: wellbore storage (compliance & q_perfs) -------------

def estimate_wellbore_compliance_from_pv(time, pressure_bar, volume_m3, mask=None):
    """
    Estimate wellbore compliance C_well [m^3/bar] from the *linear pre-breakdown* part of a PV plot.
    C_well = dV/dp (slope of V vs p).

    Parameters
    ----------
    time : array-like datetime64 (not used except to let you pass a mask in the same length)
    pressure_bar : array-like
    volume_m3 : array-like
    mask : boolean array-like or slice selecting the linear segment (optional)

    Returns
    -------
    C_well : float (m^3/bar)
    stats  : dict with {'slope','intercept','r2','n','p_range','v_range'}
    """
    p = pd.Series(pressure_bar, dtype=float)
    v = pd.Series(volume_m3, dtype=float)

    if mask is None:
        # Fallback: use the longest strictly increasing-pressure segment
        inc = p.diff().fillna(0) > 0
        mask = inc

    p_fit = p[mask].to_numpy()
    v_fit = v[mask].to_numpy()
    if len(p_fit) < 3:
        raise ValueError("Need ≥3 points from a linear pre-breakdown segment to estimate C_well.")

    # Linear regression V = a + b * p  → b = dV/dp = C_well
    A = np.vstack([np.ones_like(p_fit), p_fit]).T
    coeff, *_ = np.linalg.lstsq(A, v_fit, rcond=None)
    a, b = coeff  # intercept, slope
    v_pred = a + b * p_fit
    ss_res = np.sum((v_fit - v_pred) ** 2)
    ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    stats = {
        'slope': b, 'intercept': a, 'r2': float(r2), 'n': int(len(p_fit)),
        'p_range': (float(np.min(p_fit)), float(np.max(p_fit))),
        'v_range': (float(np.min(v_fit)), float(np.max(v_fit))),
    }
    return float(b), stats  # C_well = slope = dV/dp in m^3/bar


def _dp_dt_bar_per_s(time, pressure_bar, smooth_s=None):
    """
    Compute dp/dt in [bar/s] on a datetime axis. Optional moving-average smoothing window in seconds.
    """
    t = pd.to_datetime(pd.Series(time), errors='coerce')
    p = pd.Series(pressure_bar, dtype=float)

    # seconds from start
    ts = (t - t.iloc[0]).dt.total_seconds().to_numpy()

    # Gradient in bar/s (handles non-uniform dt)
    dpdt = np.gradient(p.to_numpy(), ts)

    if smooth_s and smooth_s > 0:
        # simple centered moving-average with ~smooth_s window
        dt_med = np.median(np.diff(ts)) if len(ts) > 1 else 1.0
        win = max(int(round(smooth_s / max(dt_med, 1e-9))), 1)
        if win > 1 and win % 2 == 0:
            win += 1  # enforce odd window
        dpdt = pd.Series(dpdt).rolling(win, center=True, min_periods=1).mean().to_numpy()

    return pd.Series(dpdt, index=p.index)


def flow_at_perfs_from_surface(q_pump, time_q, pressure_bar, time_p, C_well_m3_per_bar,
                               *, out_units='m3/h', smooth_dpdt_s=5):
    """
    q_perf(t) = q_pump(t) - C_well * (dp/dt)(t)

    Units:
      - q_pump given in m^3/h (typical surface logger) or m^3/s; auto-detect if values look small.
      - pressure in bar.
      - C_well in m^3/bar.
      - dp/dt in bar/s → C_well*dp/dt in m^3/s.

    Returns
    -------
    q_perf : pd.Series in out_units ('m3/h' or 'm3/s'), indexed like time_q.
    parts  : dict with aligned pieces for diagnostics.
    """
    # Pressure derivative on its native clock
    dpdt_bar_per_s = _dp_dt_bar_per_s(time_p, pressure_bar, smooth_s=smooth_dpdt_s)

    # Align dp/dt to the flow timestamps
    dpdt_on_q = align_series(time_p, dpdt_bar_per_s, time_q).to_numpy()

    # Convert q_pump to m^3/s if it looks like m^3/h
    qS = pd.Series(q_pump, dtype=float).to_numpy()
    if np.nanmedian(qS) > 1.0:  # crude heuristic: large → likely m^3/h
        q_pump_m3s = qS / 3600.0
        src_units = 'm3/h'
    else:
        q_pump_m3s = qS
        src_units = 'm3/s'

    # q_perf in m^3/s
    q_perf_m3s = q_pump_m3s - float(C_well_m3_per_bar) * dpdt_on_q

    if (out_units or '').lower() == 'm3/h':
        q_out = q_perf_m3s * 3600.0
    else:
        q_out = q_perf_m3s

    parts = {
        'dpdt_bar_per_s_on_q': dpdt_on_q,
        'q_pump_m3s': q_pump_m3s,
        'src_units': src_units,
    }
    return pd.Series(q_out, index=pd.Series(time_q).index), parts

def _to_epoch_s(ts):
    t = pd.to_datetime(pd.Series(ts)).astype("int64") / 1e9  # ns -> s
    return t.to_numpy(dtype=float)

# ----------------------------
# Wellbore storage & perf rate
# ----------------------------

def _dVdP_df(time_S, volume_S, pressure_S, *, small=1e-9, smooth_window=5):
    """
    Compute C_well(t)=dV/dP along with dV/dt and dP/dt on the SURFACE clock.
    Returns a DataFrame with columns:
      time, pressure_bar, volume_m3, dVdt_m3_per_s, dPdt_bar_per_s, dVdP_m3_per_bar
    """
    t_s = _to_epoch_s(time_S)
    V   = pd.Series(volume_S, dtype=float).to_numpy()
    P   = pd.Series(pressure_S, dtype=float).to_numpy()

    m = np.isfinite(t_s) & np.isfinite(V) & np.isfinite(P)
    t_s, V, P = t_s[m], V[m], P[m]
    order = np.argsort(t_s, kind="mergesort")
    t_s, V, P = t_s[order], V[order], P[order]
    t_idx = pd.to_datetime(pd.Series(time_S).to_numpy()[m][order])

    dVdt = np.gradient(V, t_s)           # m^3/s
    dPdt = np.gradient(P, t_s)           # bar/s
    dPdt_safe = dPdt.copy()
    dPdt_safe[np.abs(dPdt_safe) < small] = np.nan
    dVdP = dVdt / dPdt_safe              # m^3/bar

    if smooth_window and smooth_window >= 3:
        dVdt = pd.Series(dVdt).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
        dPdt = pd.Series(dPdt).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
        dVdP = pd.Series(dVdP).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()

    return pd.DataFrame({
        "time": t_idx,
        "pressure_bar": P,
        "volume_m3": V,
        "dVdt_m3_per_s": dVdt,
        "dPdt_bar_per_s": dPdt,
        "dVdP_m3_per_bar": dVdP
    })
