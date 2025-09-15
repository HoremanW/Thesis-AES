# plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typing import Optional, Tuple, Dict
import datetime as dt
import closure_analysis as ca

def _fmt_hhmm(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# ---------- SURFACE 2x4 GRID ----------
def plot_surface_panels(
    time_array,
    pressure_array,
    flowrate_array,
    density_array,
    volume_array,
    tot_volume_nrd_array,
    return_volume_array,
    totbalance_volume_array,
    RDA_pressure_array,
    *,
    figsize=(22, 8),
    markersize=0.2,
    linewidth=1.0,
    tick_labelsize=7,
    title_labelsize=6,
):
    """Create 2×4 panels of surface channels vs time."""
    panels = [
        (pressure_array,        'Treating Pressure [bar]', 'Treating Pressure vs Time', 'green'),
        (volume_array,          'Volume [m³]',             'Volume vs Time',            'blue'),
        (tot_volume_nrd_array,  'TOT Volume NRD [m³]',     'TOT Volume NRD vs Time',    'purple'),
        (flowrate_array,        'Flowrate [m³/h]',         'Flowrate vs Time',          'red'),
        (density_array,         'Density [kg/m³]',         'Density vs Time',           'teal'),
        (return_volume_array,   'RETURN VOL [m³]',         'RETURN VOL vs Time',        'orange'),
        (totbalance_volume_array,'Total Balance [m³]',     'Total Balance vs Time',     'purple'),
        (RDA_pressure_array,    'RDA2 IN1 [bar]',          'RDA2 IN1 vs Time',          'red'),
    ]

    fig, axs = plt.subplots(2, 4, figsize=figsize)
    font_small = {'fontsize': title_labelsize}

    for i, (y, ylabel, ptitle, color) in enumerate(panels):
        r, c = divmod(i, 4)
        ax = axs[r, c]
        ax.plot(time_array, y, color=color, marker='o',
                markersize=markersize, linestyle='-', linewidth=linewidth)
        ax.set_xlabel('Time', **font_small)
        ax.set_ylabel(ylabel, **font_small)
        ax.set_title(ptitle, **font_small)
        ax.tick_params(axis='both', labelsize=tick_labelsize)
        ax.grid(True, alpha=0.4)
        _fmt_hhmm(ax)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    return fig, axs

# ---------- TRIPLE-AXIS (PRESSURE / FLOW / RETURN VOL) ----------
def plot_triple_axis(time_array, pressure_array, flowrate_array, return_volume_array,
                     *, start=None, end=None, figsize=(16, 6)):
    font_small = {'fontsize': 8}
    fig, ax1 = plt.subplots(figsize=figsize)

    # Left: Pressure
    ax1.plot(time_array, pressure_array, linewidth=1, label='Pressure [bar]', color='royalblue')
    ax1.set_xlabel('Time', **font_small)
    ax1.set_ylabel('Pressure [bar]', **font_small)
    ax1.set_title('Pressure, Flowrate & Return Volume vs Time', **font_small)
    ax1.tick_params(axis='both', labelsize=7)
    if start is not None and end is not None:
        ax1.set_xlim(left=start, right=end)
    ax1.grid(True, alpha=0.4)

    # Right 1: Flowrate
    ax2 = ax1.twinx()
    ax2.plot(time_array, flowrate_array, linewidth=1, label='Flowrate [m³/h]', color='forestgreen')
    ax2.set_ylabel('Flowrate [m³/h]', **font_small)
    ax2.tick_params(axis='y', labelsize=7)

    # Right 2 (outer): Return volume
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(time_array, return_volume_array, linestyle='--', linewidth=1, label='Return Volume [m³]', color='darkorange')
    ax3.set_ylabel('Return Volume [m³]', **font_small)
    ax3.tick_params(axis='y', labelsize=7)

    _fmt_hhmm(ax1)

    # Unified legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3,
               fontsize=6, loc='upper center')

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)

# ---------- DOWNHOLE P/T ----------
def plot_downhole_pt(time_array2, pressure_array2, temperature_array2, *, figsize=(16, 6)):
    fig, ax4 = plt.subplots(figsize=figsize)
    ax4.plot(time_array2, pressure_array2, linewidth=1, label='Pressure [bar]', color='royalblue')
    ax4.set_xlabel('Time', fontsize=8)
    ax4.set_ylabel('Pressure [bar]', fontsize=8)
    ax4.set_title('Pressure and Temperature vs Time', fontsize=8)
    ax4.tick_params(axis='both', labelsize=7)
    ax4.grid(True, alpha=0.4)
    _fmt_hhmm(ax4)

    ax5 = ax4.twinx()
    ax5.plot(time_array2, temperature_array2, linewidth=1, label='Temperature [°C]', color='red')
    ax5.set_ylabel('Temperature [°C]', fontsize=8)
    ax5.tick_params(axis='y', labelsize=7)

    # unified legend
    l4, lb4 = ax4.get_legend_handles_labels()
    l5, lb5 = ax5.get_legend_handles_labels()
    ax4.legend(l4 + l5, lb4 + lb5, fontsize=6, loc='upper left')

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, (ax4, ax5)

# ---------- ALIGNMENT (datetime) ----------
def plot_alignment(ts_surface_dt_algn, y_surface, ts_downhole_dt, y_downhole, *, figsize=(16, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts_surface_dt_algn, y_surface, label='Surface (shifted to downhole) → fracture [bar]', linewidth=1, alpha=0.9)
    ax.plot(ts_downhole_dt,     y_downhole, label='Downhole → fracture [bar]', linewidth=1)
    ax.set_xlabel('Time (HH:MM) — downhole clock', fontsize=8)
    ax.set_ylabel('Pressure [bar]', fontsize=8)
    ax.set_title('Surface shifted to Downhole timeline (both corrected to fracture)', fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    _fmt_hhmm(ax)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

# ---------- Closure diagnostics ----------

def plot_srt(
    x_sqrt,
    p_srt,
    dpdx,
    *,
    i_cl: int | None = None,   # optional index of closure pick to mark
    cap_s: float | None = None, # cap in seconds (we’ll convert to √s for xlim)
    figsize=(10, 5),
    grid_alpha=0.5
):
    """
    Single-axes √t diagnostic:
      - Left y-axis: Pressure vs √t
      - Right y-axis: BOTH derivatives vs √t
          • normal derivative:       dP/d√t
          • semilog derivative:      √t · dP/d√t

    Parameters
    ----------
    x_sqrt : array-like
        √t values (seconds^0.5).
    p_srt : array-like
        Pressure values aligned to x_sqrt.
    dpdx : array-like
        dP/d√t aligned to x_sqrt.
    i_cl : int or None
        Optional index to draw a vertical marker at x_sqrt[i_cl].
    cap_s : float or None
        If provided, sets x-limits to [0, sqrt(cap_s)].
    """

    x = np.asarray(x_sqrt, float)
    p = np.asarray(p_srt,  float)
    d = np.asarray(dpdx,   float)

    # Semilog derivative in √t-domain (Barree/Nolte convention)
    d_semilog = x * d  # √t · dP/d√t

    fig, ax_left = plt.subplots(figsize=figsize)

    # Left axis: pressure vs √t
    line_p, = ax_left.plot(x, p, lw=1.2, color="tab:blue", label="Pressure vs √t")
    ax_left.set_xlabel("√t [s$^{0.5}$]")
    ax_left.set_ylabel("Pressure [bar]", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.grid(True, ls="--", lw=0.7, alpha=grid_alpha)

    # Right axis: both derivatives vs √t on the SAME scale
    ax_right = ax_left.twinx()
    line_d_norm,   = ax_right.plot(x, d,          lw=1.2, ls="-.", color="tab:orange", label="dP/d√t")
    line_d_semi,   = ax_right.plot(x, d_semilog,  lw=1.2, ls="--", color="tab:red",    label="√t·dP/d√t")
    ax_right.set_ylabel("Derivatives [bar s$^{-0.5}$] / [bar]", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    # Auto-scale right y-axis to the data range (both derivatives together)
    data_right = np.concatenate([d.reshape(-1), d_semilog.reshape(-1)])
    data_right = data_right[np.isfinite(data_right)]
    if data_right.size:
        ymin = float(np.nanmin(data_right))
        ymax = float(np.nanmax(data_right))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.1 * (ymax - ymin)
            ax_right.set_ylim(ymin - pad, ymax + pad)
    
    # Optional cap on x
    if cap_s is not None and np.isfinite(cap_s):
        ax_left.set_xlim(0.0, float(np.sqrt(cap_s)))

    # Optional vertical marker at closure pick
    if i_cl is not None and 0 <= i_cl < len(x) and np.isfinite(x[i_cl]):
        ax_left.axvline(x[i_cl], ls=":", lw=1.2, color="k")

    # Combined legend (left + right)
    lines = [line_p, line_d_norm, line_d_semi]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="best")

    fig.tight_layout()
    return fig, (ax_left, ax_right)



def plot_bourdet(t_log, dP_dlogt, *, p=None, p_times=None, i_cl=None, cap_s=None, figsize=(10, 5)):
    """
    Bourdet diagnostic with two y-axes:
      - Left y-axis : Pressure vs time (interpolated to t_log if needed)
      - Right y-axis: Bourdet derivative dP/dln(t)

    Parameters
    ----------
    t_log : array-like
        Time since shut-in [s] (x-axis of both curves).
    dP_dlogt : array-like
        Bourdet derivative values (same length as t_log).
    p : array-like or None
        Pressure samples. If p_times is provided and lengths differ from t_log,
        p will be interpolated onto t_log.
    p_times : array-like or None
        Timebase (in seconds since shut-in) for `p`. Required if len(p) != len(t_log).
    i_cl : int or None
        Index (in t_log) to draw a vertical closure marker.
    cap_s : float or None
        If provided, sets x-limits to [0, cap_s].
    """
    # x-axis and derivative
    t = np.asarray(t_log, float)
    d = np.asarray(dP_dlogt, float)

    fig, ax_left = plt.subplots(figsize=figsize)

    # --- Left axis: Pressure vs time (aligned to t_log) ---
    line_p = None
    if p is not None:
        p_arr = np.asarray(p, float)

        if p_times is not None and len(p_arr) != len(t):
            # Interpolate pressure to t_log
            pt = np.asarray(p_times, float)
            m = np.isfinite(pt) & np.isfinite(p_arr)
            if m.any():
                # sort by time for np.interp
                order = np.argsort(pt[m])
                pt_sorted = pt[m][order]
                pp_sorted = p_arr[m][order]
                # clip t to interpolation domain to avoid NaNs
                t_clip = np.clip(t, pt_sorted[0], pt_sorted[-1])
                p_interp = np.interp(t_clip, pt_sorted, pp_sorted)
                p_plot = p_interp
            else:
                p_plot = np.full_like(t, np.nan, dtype=float)
        else:
            # lengths match or we just trust p is already on t
            p_plot = p_arr if len(p_arr) == len(t) else p_arr[:len(t)]

        line_p, = ax_left.plot(t, p_plot, lw=1.2, color="tab:blue", label="Pressure")
        ax_left.set_ylabel("Pressure [bar]", color="tab:blue")
        ax_left.tick_params(axis="y", labelcolor="tab:blue")

    ax_left.set_xlabel("Time since shut-in [s]")
    ax_left.grid(True, ls="--", lw=0.7, alpha=0.5)

    # --- Right axis: Bourdet derivative ---
    ax_right = ax_left.twinx()
    line_d, = ax_right.plot(t, d, lw=1.2, ls="--", color="tab:red", label="dP/dln(t)")
    ax_right.set_ylabel("dP/dln(t) [bar]", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    # Optional x limit
    if cap_s is not None:
        ax_left.set_xlim(0, float(cap_s))

    # Optional closure marker
    if i_cl is not None and 0 <= i_cl < len(t) and np.isfinite(t[i_cl]):
        ax_left.axvline(t[i_cl], ls=":", lw=1.2, color="k")

    # Combined legend
    lines = [l for l in (line_p, line_d) if l is not None]
    if lines:
        ax_left.legend(lines, [l.get_label() for l in lines], loc="best")

    fig.tight_layout()
    return fig, ax_left

# assumes closure_analysis.g_function_high_efficiency and .semilog_derivative exist

def plot_gfunction(
    ts_seconds,
    p,
    tp_seconds,
    *,
    p_times=None,
    i_cl=None,
    cap_s=None,
    figsize=(10, 5)
):
    """
    G-function diagnostic with two y-axes:
      - Left y-axis : Pressure vs G(ts)
      - Right y-axis: Semilog derivative (G·dP/dG) AND normal derivative (dP/dG)

    Parameters
    ----------
    ts_seconds : array-like
        Time since shut-in [s] used to build G.
    p : array-like
        Pressure samples. If `p_times` is given and len(p) != len(ts_seconds),
        p will be interpolated onto ts_seconds before computing the derivative.
    tp_seconds : float
        Pumping time [s] (t_shut - t_pump_start) for the cycle.
    p_times : array-like or None
        Time base [s] for `p` if it differs from `ts_seconds`.
    i_cl : int or None
        Index (in ts_seconds) to draw a vertical marker (both axes).
    cap_s : float or None
        If provided, x-limit will be set to G at min(cap_s, ts_seconds.max()).
    """
    from closure_analysis import g_function_high_efficiency, semilog_derivative

    t = np.asarray(ts_seconds, float)
    p_arr = np.asarray(p, float)

    # Align pressure to the time grid if needed (same logic as Bourdet helper)
    if p_times is not None and len(p_arr) != len(t):
        pt = np.asarray(p_times, float)
        m = np.isfinite(pt) & np.isfinite(p_arr)
        if m.any():
            order = np.argsort(pt[m])
            pt_sorted = pt[m][order]
            pp_sorted = p_arr[m][order]
            t_clip = np.clip(t, pt_sorted[0], pt_sorted[-1])
            p_plot = np.interp(t_clip, pt_sorted, pp_sorted)
        else:
            p_plot = np.full_like(t, np.nan, dtype=float)
    else:
        p_plot = p_arr if len(p_arr) == len(t) else p_arr[:len(t)]

    # Compute G
    G = g_function_high_efficiency(t, float(tp_seconds))

    # --- Derivatives ---
    # Semilog derivative (Barree plotting convention)
    semilog_dP = semilog_derivative(G, p_plot)

    # Normal derivative dP/dG (central differences; one-sided at ends)
    dP_dG = np.full_like(p_plot, np.nan, dtype=float)
    if len(G) >= 2:
        # one-sided ends
        dG = G[1] - G[0]
        if dG != 0:
            dP_dG[0] = (p_plot[1] - p_plot[0]) / dG
        dG = G[-1] - G[-2]
        if dG != 0:
            dP_dG[-1] = (p_plot[-1] - p_plot[-2]) / dG
    if len(G) >= 3:
        for i in range(1, len(G) - 1):
            dG = G[i + 1] - G[i - 1]
            if dG != 0:
                dP_dG[i] = (p_plot[i + 1] - p_plot[i - 1]) / dG

    # --- Figure ---
    fig, ax_left = plt.subplots(figsize=figsize)

    # Left axis: P vs G
    line_p, = ax_left.plot(G, p_plot, lw=1.2, color="tab:blue", label="Pressure vs G")
    ax_left.set_xlabel("G-function (dimensionless)")
    ax_left.set_ylabel("Pressure [bar]", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.grid(True, ls="--", lw=0.7, alpha=0.5)

    # Right axis: both derivatives vs G (same right scale)
    ax_right = ax_left.twinx()
    line_semilog, = ax_right.plot(G, semilog_dP, lw=1.2, ls="--", color="tab:red", label="Semilog derivative (G·dP/dG)")
    line_normal,  = ax_right.plot(G, dP_dG,      lw=1.2, ls="-.", color="tab:orange", label="Normal derivative (dP/dG)")
    ax_right.set_ylabel("Derivatives [bar]", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    # Optional x-limit using time cap converted to G cap
    if cap_s is not None and np.isfinite(cap_s):
        t_cap = min(float(cap_s), float(np.nanmax(t)) if len(t) else float(cap_s))
        idx_cap = np.where(t <= t_cap)[0]
        if len(idx_cap):
            ax_left.set_xlim(G[idx_cap[0]], G[idx_cap[-1]])

    # Optional vertical marker (index in ts array)
    if i_cl is not None and 0 <= i_cl < len(G) and np.isfinite(G[i_cl]):
        ax_left.axvline(G[i_cl], ls=":", lw=1.2, color="k")

    # Combined legend
    lines = [line_p, line_semilog, line_normal]
    ax_left.legend(lines, [l.get_label() for l in lines], loc="best")
    #ax_right.set_ylim(-10, 10)

    fig.tight_layout()
    return fig, ax_left



def _central_diff(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.full_like(x, np.nan, dtype=float)
    dy = np.empty_like(y); dx = np.empty_like(x)
    dx[1:-1] = x[2:] - x[:-2]; dy[1:-1] = y[2:] - y[:-2]
    dx[0] = x[1] - x[0]; dx[-1] = x[-1] - x[-2]
    dy[0] = y[1] - y[0]; dy[-1] = y[-1] - y[-2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return dy / dx

def _auto_closure_idx_semilog(x, y_semilog, fit_frac=0.25, tol=0.10):
    """
    Fit straight line through origin using early portion; first sustained departure => closure.
    """
    n = len(x)
    k = max(5, int(fit_frac * n))
    X = x[:k]; Y = y_semilog[:k]
    m = float(np.dot(X, Y) / max(1e-12, np.dot(X, X)))  # slope through origin
    baseline = m * x
    diff = np.abs(y_semilog - baseline)
    thresh = np.maximum(1e-9, tol * np.abs(baseline))
    window = 3
    for i in range(window, n):
        if np.all(diff[i-window+1:i+1] > thresh[i-window+1:i+1]):
            return i
    return None

def barree_prepare(time_seconds_since_shutin, pressure, tp_seconds, isip=None):
    """
    Prepare Δt, ΔP, G, √t arrays from a falloff series p(t) that *starts at shut-in*.
    """
    t = np.asarray(time_seconds_since_shutin, float)
    p = np.asarray(pressure, float)
    assert len(t) == len(p) and len(t) >= 3, "need aligned post-shut-in series"
    if isip is None:
        isip = float(p[0])
    dP = p - float(isip)
    G = ca.g_function_high_efficiency(t, tp_seconds)
    S = np.sqrt(t)
    return t, p, dP, G, S, float(isip)

def plot_fig1_G(G, p, closure_idx: Optional[int] = None, *, figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize)
    dPdG = _central_diff(G, p)
    GdPdG = G * dPdG
    ax.plot(G, p, label='P vs. G')
    ax.set_xlabel('G (Time)'); ax.set_ylabel('Pressure'); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(G, GdPdG, ls='--', lw=1.3, label='G·dP/dG')
    ax2.plot(G, dPdG,  ls=':',  lw=1.3, label='dP/dG')
    ax2.set_ylabel('Derivatives')
    if closure_idx is not None and 0 <= closure_idx < len(G):
        ax.axvline(G[closure_idx], color='k', ls='--'); ax.text(G[closure_idx], ax.get_ylim()[1]*0.98, 'Fracture Closure', va='top')
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax2.legend(l1+l2, lb1+lb2, loc='upper right')
    fig.tight_layout(); return fig

def plot_fig2_sqrt_t(S, p, closure_idx: Optional[int] = None, *, figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize)
    dPdS = _central_diff(S, p); SdPdS = S * dPdS
    ax.plot(S, p, label='P vs. √t'); ax.set_xlabel('√(shut-in time)'); ax.set_ylabel('Pressure'); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(S, SdPdS, ls='--', lw=1.3, label='√t·dP/d√t')
    ax2.plot(S, dPdS,  ls=':',  lw=1.3, label='dP/d√t')
    ax2.set_ylabel('Derivatives')
    if closure_idx is not None and 0 <= closure_idx < len(S):
        ax.axvline(S[closure_idx], color='k', ls='--'); ax.text(S[closure_idx], ax.get_ylim()[1]*0.98, 'Fracture Closure', va='top')
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax2.legend(l1+l2, lb1+lb2, loc='upper right')
    fig.tight_layout(); return fig

def plot_fig3_loglog_deltaP(t, dP, closure_idx: Optional[int] = None, *, figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(t, float); y = np.asarray(dP, float)
    y_semilog = ca.semilog_derivative(x, y)  # Δt * dΔP/dΔt
    ax.loglog(np.clip(x, 1e-9, None), np.abs(y), label='ΔP vs. Δt')
    ax.loglog(np.clip(x, 1e-9, None), np.abs(y_semilog), ls='--', label='Δt·dΔP/dΔt')
    ax.set_xlabel('Shut-in time, Δt'); ax.set_ylabel('ΔP and derivative'); ax.grid(True, which='both', ls=':', alpha=0.4)
    if closure_idx is not None and 0 <= closure_idx < len(x):
        ax.axvline(max(x[closure_idx], 1e-9), color='k', ls='--'); ax.text(max(x[closure_idx], 1e-9), ax.get_ylim()[1]*0.8, 'Fracture Closure', va='top')
    ax.legend(loc='best'); fig.tight_layout(); return fig

def _FL2(t, tc):
    t = np.asarray(t, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = (t - tc) / max(1e-12, tc)
    return np.clip(X, 1e-12, None)

def plot_fig4_ACA_log(fl2, dP, pi_guess: Optional[float] = None, *, figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize)
    Y = np.asarray(dP, float) if pi_guess is None else (np.asarray(dP, float) - float(pi_guess))
    der = ca.semilog_derivative(fl2, Y)  # FL² * dΔP/d(FL²)
    ax.loglog(fl2, np.abs(Y), label='ΔP vs. FL²')
    ax.loglog(fl2, np.abs(der), ls='--', label='FL²·dΔP/d(FL²)')
    ax.set_xlabel('Square Linear Flow (FL²)'); ax.set_ylabel('ΔP and derivative'); ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(loc='best'); fig.tight_layout(); return fig

def plot_barree_figs_normal_leakoff(
    t_post_s: np.ndarray,
    p_post: np.ndarray,
    tp_seconds: float,
    *,
    isip: Optional[float] = None,
    closure_idx: Optional[int] = None,
    auto_closure: bool = True,
    auto_fit_frac: float = 0.25,
    auto_tol: float = 0.10,
    closure_time_s: Optional[float] = None,
    pi_guess: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Build Figures 1–4 in one call. Inputs are *post-shut-in* arrays on the downhole clock.
    - tp_seconds is the pumping (fracture-extension) duration.
    - If auto_closure=True, we pick closure from the semilog derivative on G per Barree.
    """
    t, p, dP, G, S, ISIP = barree_prepare(t_post_s, p_post, tp_seconds, isip=isip)

    if closure_idx is None and auto_closure:
        # semilog derivative on G (G * dP/dG) should be a straight line through origin pre-closure
        dPdG = _central_diff(G, p); GdPdG = G * dPdG
        closure_idx = _auto_closure_idx_semilog(G, GdPdG, fit_frac=auto_fit_frac, tol=auto_tol)

    if closure_time_s is None and closure_idx is not None:
        closure_time_s = float(t[closure_idx])
    if closure_time_s is None:
        closure_time_s = max(1.0, float(np.nanmin(t)))

    figs = {}
    figs['barree_fig1_G']        = plot_fig1_G(G, p, closure_idx=closure_idx)
    figs['barree_fig2_sqrt']     = plot_fig2_sqrt_t(S, p, closure_idx=closure_idx)
    figs['barree_fig3_loglog']   = plot_fig3_loglog_deltaP(t, dP, closure_idx=closure_idx)
    figs['barree_fig4_ACA_log']  = plot_fig4_ACA_log(_FL2(t, closure_time_s), dP, pi_guess=pi_guess)
    return figs

# Plots matching Barree et al. (2009) Figures 1–4 for the "normal leakoff" example


def _extract_line_from_result(res, side, x_default):
    """
    Helper: pull a fit line for 'left' or 'right' from the result dict, in a robust way.

    Supports either:
      - arrays: res[f'{side}_x'], res[f'{side}_y']
      - or slope/intercept + span: res[f'm_{side}'], res[f'b_{side}'], and optionally
        res[f'{side}_x_span'] (2-tuple) to build a line over that domain.
    Falls back to using the full x_default span if no span info is available.
    """
    # direct arrays have priority if present
    x_key = f'{side}_x'
    y_key = f'{side}_y'
    if x_key in res and y_key in res:
        x = np.asarray(res[x_key], float)
        y = np.asarray(res[y_key], float)
        if len(x) and len(y) and len(x) == len(y):
            return x, y

    # slope/intercept path
    m_key = f'm_{side}'
    b_key = f'b_{side}'
    m = res.get(m_key, None)
    b = res.get(b_key, None)
    if m is None or b is None:
        return None, None

    # choose domain
    span_key = f'{side}_x_span'
    if span_key in res and res[span_key] is not None:
        try:
            x0, x1 = res[span_key]
            x_line = np.linspace(float(x0), float(x1), 50)
        except Exception:
            x_line = np.linspace(np.nanmin(x_default), np.nanmax(x_default), 50)
    else:
        x_line = np.linspace(np.nanmin(x_default), np.nanmax(x_default), 50)

    y_line = m * x_line + b
    return x_line, y_line


def plot_srt_intersection(
    ts_seconds,
    p,
    res_srt,
    *,
    cap_s=180,
    figsize=(9, 5),
    grid_alpha=0.45
):
    """
    Plot √t intersection result on a single axes:
      - x-axis: √t
      - y-axis (left): Pressure
      - overlays: left/right fit lines and the intersection marker.

    Expects res_srt to contain at least:
      - 'ok' (bool), 'closure_time_s' (float), 'closure_pressure_bar' (float)
    Optionally one of:
      - arrays: 'left_x','left_y','right_x','right_y' (in √t space)
      - or lines: 'm_left','b_left','m_right','b_right' and optional spans
                  'left_x_span','right_x_span' (in √t space)
    """
    t = np.asarray(ts_seconds, float)
    P = np.asarray(p, float)
    x = np.sqrt(t)

    # Cap to first cap_s seconds
    if cap_s is not None and np.isfinite(cap_s):
        keep = np.isfinite(x) & np.isfinite(P) & (t >= 0) & (t <= float(cap_s))
        x, P = x[keep], P[keep]
    else:
        keep = np.isfinite(x) & np.isfinite(P) & (t >= 0)
        x, P = x[keep], P[keep]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, P, lw=1.2, color="tab:blue", label="Pressure vs √t")
    ax.set_xlabel("√t [s$^{0.5}$]")
    ax.set_ylabel("Pressure [bar]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, ls="--", lw=0.7, alpha=grid_alpha)

    # Draw intersection lines if available
    Lx, Ly = _extract_line_from_result(res_srt, 'left',  x)
    Rx, Ry = _extract_line_from_result(res_srt, 'right', x)
    if Lx is not None and Ly is not None:
        ax.plot(Lx, Ly, color="tab:orange", lw=1.2, ls="-.", label="Left fit")
    if Rx is not None and Ry is not None:
        ax.plot(Rx, Ry, color="tab:green",  lw=1.2, ls="--", label="Right fit")

    # Intersection marker
    if res_srt.get('ok', False):
        t_star = float(res_srt.get('closure_time_s', np.nan))
        p_star = float(res_srt.get('closure_pressure_bar', np.nan))
        if np.isfinite(t_star) and np.isfinite(p_star):
            x_star = np.sqrt(max(0.0, t_star))
            ax.plot([x_star], [p_star], 'ko', ms=5, label="√t intersection")

    # x-limits from cap
    if cap_s is not None and np.isfinite(cap_s):
        ax.set_xlim(0, np.sqrt(float(cap_s)))

    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_g_intersection(
    ts_seconds,
    p,
    res_g,
    *,
    tp_seconds,
    cap_s=180,
    figsize=(9, 5),
    grid_alpha=0.45
):
    """
    Plot G-function intersection result on a single axes:
      - x-axis: G(t)
      - y-axis: Pressure
      - overlays: left/right fit lines (in G-domain) and intersection marker.

    Expects res_g to contain at least:
      - 'ok' (bool), 'closure_time_s' (float), 'closure_pressure_bar' (float)
    Optionally one of:
      - arrays: 'left_x','left_y','right_x','right_y' (in G space)
      - or lines: 'm_left','b_left','m_right','b_right' and optional spans
                  'left_x_span','right_x_span' (in G space)
    """
    from closure_analysis import g_function_high_efficiency

    t = np.asarray(ts_seconds, float)
    P = np.asarray(p,          float)

    # Cap to first cap_s seconds
    if cap_s is not None and np.isfinite(cap_s):
        keep = np.isfinite(t) & np.isfinite(P) & (t >= 0) & (t <= float(cap_s))
        t, P = t[keep], P[keep]
    else:
        keep = np.isfinite(t) & np.isfinite(P) & (t >= 0)
        t, P = t[keep], P[keep]

    # Build G on the same subset
    G = g_function_high_efficiency(t, float(tp_seconds))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(G, P, lw=1.2, color="tab:blue", label="Pressure vs G")
    ax.set_xlabel("G-function (dimensionless)")
    ax.set_ylabel("Pressure [bar]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, ls="--", lw=0.7, alpha=grid_alpha)

    # Draw intersection lines if available (in G-domain)
    Lx, Ly = _extract_line_from_result(res_g, 'left',  G)
    Rx, Ry = _extract_line_from_result(res_g, 'right', G)
    if Lx is not None and Ly is not None:
        ax.plot(Lx, Ly, color="tab:orange", lw=1.2, ls="-.", label="Left fit")
    if Rx is not None and Ry is not None:
        ax.plot(Rx, Ry, color="tab:green",  lw=1.2, ls="--", label="Right fit")

    # Intersection marker
    if res_g.get('ok', False):
        t_star = float(res_g.get('closure_time_s', np.nan))
        p_star = float(res_g.get('closure_pressure_bar', np.nan))
        if np.isfinite(t_star) and np.isfinite(p_star):
            # Convert t* to G*
            G_star = g_function_high_efficiency(np.array([t_star], float), float(tp_seconds))[0]
            ax.plot([G_star], [p_star], 'ko', ms=5, label="G intersection")

    # If you want to cap x by time cap, convert cap to a G cap
    if cap_s is not None and np.isfinite(cap_s) and len(G):
        # domain already capped by t in the mask; xlim can simply span that G range
        ax.set_xlim(np.nanmin(G), np.nanmax(G))

    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

