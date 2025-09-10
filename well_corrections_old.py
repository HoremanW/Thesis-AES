import numpy as np
import pandas as pd

g = 9.80665  # gravitational acceleration [m/s^2]

def series_to_seconds(t):
    """
    Convert a datetime-like pandas Series to relative seconds from its first timestamp.

    Parameters
    ----------
    t : array-like or pd.Series of datetime64[ns]
        Time stamps (need to be coercible to a pandas Series of datetimes).

    Returns
    -------
    np.ndarray
        1D array of floats: seconds elapsed since t.iloc[0].

    Notes
    -----
    - This creates a *local* relative time axis starting at zero. It does NOT
      convert to absolute epoch seconds.
    - If you need two series on a *shared* grid, convert each to seconds with
      its own zero, then build a common grid (as the calling code does).
    """
    t = pd.Series(t)
    return (t - t.iloc[0]).dt.total_seconds().to_numpy()


def align_series(src_time, src_vals, dst_time, kind='linear'):
    """
    Interpolate (or step) values defined on `src_time` onto `dst_time`.

    Parameters
    ----------
    src_time : array-like or pd.Series of datetime64[ns]
        Timestamps for the source values.
    src_vals : array-like (numeric)
        Source values measured at `src_time`.
    dst_time : array-like or pd.Series of datetime64[ns]
        Target timestamps to which you want to map/interpolate the source values.
    kind : str, optional
        Interpolation mode. Currently only 'linear' is implemented.
        (The parameter is kept for future extensibility.)

    Returns
    -------
    pd.Series
        Values aligned to `dst_time`. The returned Series uses the *index* of
        `dst_time` to make it easy to keep downstream alignment.

    Behavior & Assumptions
    ----------------------
    - Time handling: converts both `src_time` and `dst_time` to *relative seconds*
      (each w.r.t. its own start). This is robust to absolute timestamp offsets
      and makes it easy to resample onto a common uniform grid elsewhere.
    - Monotonicity: if `src_time` contains non-increasing steps (duplicates or
      out-of-order stamps), those samples are dropped to enforce a strictly
      increasing time base for interpolation.
    - Edges: `np.interp` is used with `left=src_vals[0]` and `right=src_vals[-1]`,
      so the aligned series is *edge-filled* with the nearest endpoint value
      outside the convex hull of `src_time`.
    - Units: no unit conversions are performed here; it’s a pure resampling step.
    """
    ts_src = series_to_seconds(src_time)
    ts_dst = series_to_seconds(dst_time)

    # Drop non-increasing timestamps to ensure a strictly increasing axis
    keep = np.concatenate(([True], np.diff(ts_src) > 0))
    ts_src = ts_src[keep]
    src_vals = pd.Series(src_vals).to_numpy()[keep]

    # Linear interpolation. Endpoints use nearest edge values.
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
    Apply a hydrostatic correction to downhole gauge pressure to estimate pressure at the fracture depth.

    Concept
    -------
    If your gauge is installed shallower than the fracture, the fracture pressure is
    higher by the hydrostatic head of the fluid column between those vertical depths.

    P_fracture(t) = P_gauge(t) + ρ(t) * g * ΔTVD

    where ΔTVD = TVD_fracture - TVD_gauge (positive if fracture is deeper).

    Parameters
    ----------
    p_gauge : array-like (numeric)
        Downhole gauge pressure series measured at `time_gauge`.
        IMPORTANT: `p_gauge` must already be in the same units you request via `out_units`
        because the hydrostatic increment will be converted to that unit before addition.
    time_gauge : array-like or pd.Series of datetime64[ns]
        Timestamps for `p_gauge`.
    rho_surface : array-like (numeric)
        Fluid density series (typically from surface logger). Can be in kg/m^3 or g/cm^3.
        A simple heuristic converts g/cm^3 → kg/m^3 when the median looks ~1.0–1.2.
    time_surface : array-like or pd.Series of datetime64[ns]
        Timestamps for `rho_surface`.
    delta_tvd_m : float
        Vertical depth difference [m], ΔTVD = TVD_fracture - TVD_gauge.
        Use TVD (true vertical depth), NOT MD.
    out_units : {'bar','mpa'} or anything else (treated as Pa), default 'bar'
        Unit for the *returned* hydrostatic increment and corrected pressure.
        - 'bar' → returns in bar
        - 'mpa' → returns in MPa
        - anything else → returns in Pa
        Make sure `p_gauge` is in the same unit as `out_units` for a meaningful sum.
    lag_s : float or int, optional
        Known time lag in seconds between the *downhole* gauge and the *surface* density:
        - If `lag_s > 0`: downhole (gauge) lags surface → we shift the gauge times *back*
          by `lag_s` to align density(t) with the corresponding gauge sample.
        - If `lag_s < 0`: downhole leads surface → shift gauge times *forward* (negative shift).

    Returns
    -------
    p_at_fracture : pd.Series
        Gauge pressure corrected to the fracture depth, in `out_units`.
        Indexed like `time_gauge` for easy plotting/merging.
    dP_hydro : pd.Series
        The hydrostatic increment ρ g ΔTVD at each timestamp, in `out_units`.

    Practical Notes
    ---------------
    - Depths: Always compute ΔTVD with *true vertical depth* (mTVDRT), not MD/MDRT.
      If you only have MD + survey, compute TVD using minimum curvature.
    - Density: If you have a better downhole density model (e.g., PVT with T, P), use it.
      This function simply aligns whatever density you provide to the gauge times.
    - Friction: During injection/flow, there may be additional friction losses between
      the gauge and fracture. This function does NOT include friction; it’s a pure
      hydrostatic correction. A Darcy–Weisbach term can be added if you provide
      flow path, length, ID, roughness, and fluid viscosity.
    - Edges and alignment: `rho_surface` is interpolated to the (possibly shifted)
      gauge time base using `align_series`. Outside the surface-density time span,
      values are edge-filled with nearest endpoint density.
    """

    
    # Optionally shift the gauge times so density(t) aligns with the correct physical instant
    tg = pd.Series(time_gauge)
    if lag_s is not None:
        tg = tg - pd.to_timedelta(lag_s, unit='s')

    # Convert/clean density and map it onto the (possibly shifted) gauge time base
    rho = pd.Series(rho_surface).astype(float)

    # Heuristic unit fix: if density looks like g/cm^3 (~1.0), convert to kg/m^3.
    # Median < 20 is a safe threshold to detect g/cm^3 without misclassifying light brines.
    if rho.dropna().median() < 20:
        rho = rho * 1000.0  # g/cm^3 → kg/m^3

    # Interpolate density to gauge timestamps
    rho_on_gauge = align_series(time_surface, rho, tg)

    # Hydrostatic increment at each timestamp: ρ g ΔTVD  [Pa]
    dP_Pa = rho_on_gauge.to_numpy() * g * float(delta_tvd_m)

    # Convert hydrostatic increment to requested output units
    out = (out_units or '').lower()
    if out == 'bar':
        dP = dP_Pa / 1e5
    elif out == 'mpa':
        dP = dP_Pa / 1e6
    else:
        dP = dP_Pa  # Pa

    # Ensure gauge pressure is numeric (AND already in the same units as `out_units`)
    p_gauge = pd.Series(p_gauge).astype(float)

    # Corrected pressure at fracture depth
    p_at_fracture = p_gauge + dP

    # Return Series aligned with the *original* gauge time index (unshifted),
    # which is typically what you want to plot against.
    return (
        pd.Series(p_at_fracture, index=pd.Series(time_gauge).index),
        pd.Series(dP,           index=pd.Series(time_gauge).index),
    )

# ---------- Optional helper if/when you want friction as well ----------

def darcy_weisbach_friction_drop(
    rho,               # kg/m^3
    Q,                 # volumetric flow rate [m^3/s]
    L,                 # flow-path length [m]
    path='tubing',     # 'tubing' or 'annulus'
    D=None,            # pipe ID [m] if path='tubing'
    Do=None, Di=None,  # annulus: outer and inner diameters [m]
    mu=None,           # dynamic viscosity [Pa·s] (needed if f not given)
    roughness=0.0,     # absolute roughness ε [m]
    f=None             # Darcy friction factor (if provided, skips Re/roughness calc)
):
    """
    Compute frictional pressure loss via Darcy–Weisbach:

        ΔP_fric = f · (L/D_h) · (ρ v² / 2)

    where:
      - v = Q / A is the cross-sectional average velocity,
      - D_h = D (pipe) or D_o - D_i (annulus hydraulic diameter),
      - f is the Darcy friction factor (NOT Fanning).

    If `f` is not provided, it is estimated:
      - Laminar (Re < 2300): f = 64 / Re
      - Turbulent: Swamee–Jain explicit correlation
          f = 0.25 / [ log10( (ε/(3.7 D_h)) + 5.74 / Re^0.9 ) ]^2

    Returns
    -------
    dP_pa : float
        Frictional pressure drop [Pa].
    out : dict
        Diagnostic info: {'A': area, 'Dh': hydraulic_diameter, 'v': velocity, 'Re': reynolds, 'f': friction_factor}

    Notes
    -----
    - Units: Q must be in m^3/s (convert from m^3/h by dividing by 3600).
    - For annulus: A = π/4 (D_o^2 − D_i^2), D_h = D_o − D_i.
    - For tubing:  A = π D^2 / 4,       D_h = D.
    - This returns ΔP along *L* in the flow direction. Add this to the hydrostatic
      correction if you want the total gauge-to-fracture correction during injection.
    """
    # Geometry
    if path.lower() == 'annulus':
        if Do is None or Di is None:
            raise ValueError("For annulus, provide Do and Di (in meters).")
        A = 0.25 * np.pi * (Do**2 - Di**2)
        Dh = Do - Di
    else:
        if D is None:
            raise ValueError("For tubing, provide D (ID in meters).")
        A = 0.25 * np.pi * D**2
        Dh = D

    # Velocity
    v = Q / A

    # Friction factor
    Re = None
    if f is None:
        if mu is None:
            raise ValueError("Provide mu (Pa·s) or directly provide f (Darcy).")
        Re = abs(rho * v * Dh / mu)
        if Re < 2300 and Re > 0:
            f = 64.0 / Re
        else:
            # Swamee–Jain explicit correlation for turbulent regimes
            eps = max(roughness, 0.0)
            f = 0.25 / (np.log10((eps / (3.7 * Dh)) + (5.74 / (Re**0.9))))**2

    # Pressure drop
    dP_pa = f * (L / Dh) * 0.5 * rho * v**2
    return float(dP_pa), {'A': A, 'Dh': Dh, 'v': v, 'Re': Re, 'f': f}

def estimate_lag(TVD_fracture_m, MD, TVD, gauge_index):
    """
    Estimate the lag (ΔTVD) between a gauge and a fracture depth.

    Parameters
    ----------
    MD : array-like
        Measured Depth values (must be same length as TVD).
    TVD : array-like
        True Vertical Depth values; can contain NaN for missing values.
    gauge_index : int
        Index of the gauge depth in MD/TVD arrays.
    TVD_fracture_m : float
        Known TVD of the fracture depth (in meters).

    Returns
    -------
    dict with:
        - TVD_interp : np.ndarray, TVD values with NaNs filled by interpolation.
        - TVD_gauge_m : float, interpolated TVD at the gauge MD.
        - delta_tvd_m : float, difference between fracture TVD and gauge TVD.
    """
    MD = np.asarray(MD, dtype=float)
    TVD = np.asarray(TVD, dtype=float)

    # Interpolate missing TVD values
    TVD_interp = np.interp(MD, MD[~np.isnan(TVD)], TVD[~np.isnan(TVD)])

    # Gauge TVD at the given index
    TVD_gauge_m = TVD_interp[gauge_index]

    # Delta TVD
    delta_tvd_m = TVD_fracture_m - TVD_gauge_m

    return TVD_interp, TVD_gauge_m, delta_tvd_m
    

