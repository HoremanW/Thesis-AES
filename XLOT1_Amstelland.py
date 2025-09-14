from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# Local modules
import time_difference
import well_corrections
import closure_analysis_old as closure_analysis
import plotting

# ----------------------------
# File paths (cross-platform)
# ----------------------------
base = Path("Data") / "Amstelland"

SURF_PATH = base / "NLOG_GS_PUB_AMS-01_XLOT 1 - Main Claystone_Cementing Unit_Data.txt"
DOWN_PATH = base / "NLOG_GS_PUB_AMS-01_XLOT 1 - Main Claystone_Downhole Gauge_Data.TXT"

# ----------------------------
# Helpers
# ----------------------------
def to_num(series):
    """
    Normalize numeric text: strip whitespace, remove NBSPs, comma→dot, coerce to float.
    Returns a pandas Series[float].
    """
    s = pd.Series(series, copy=False)
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip()
    s = s.str.replace('\u00A0', '', regex=False)   # NBSP
    s = s.str.replace(',', '.', regex=False)       # decimal comma -> dot
    s = s.replace({'': None})
    return pd.to_numeric(s, errors='coerce')

def get_or_nan(df, col):
    """Return numeric series for existing column or a NaN series matching df length."""
    if col in df.columns:
        return to_num(df[col])
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


# ----------------------------
# Example usage
# ----------------------------
dfS = pd.read_csv(SURF_PATH, sep="\t", engine="python", on_bad_lines="skip")
dfD = pd.read_csv(DOWN_PATH, sep="\t", engine="python", on_bad_lines="skip")

print("Surface data file loaded:", SURF_PATH.resolve())
print("Downhole data file loaded:", DOWN_PATH.resolve())

# Ensure expected columns exist for downhole file
dfD.columns = ['Time', 'Delta Time', 'Pressure', 'Temperature', 'None', 'None2']

# Slice away headers/units lines typical of these exports
dfS = dfS.iloc[1:].reset_index(drop=True)
dfD = dfD.iloc[4:, :-2].reset_index(drop=True)  # drop last two 'None' cols

# ----------------------------
# Parse time columns
# ----------------------------
# Surface time format: %m:%d:%Y:%H:%M:%S
time_surface = pd.to_datetime(dfS['Time'], format='%m:%d:%Y:%H:%M:%S', errors='coerce')

# Downhole time format: %d-%m-%Y %H:%M:%S
time_downhole = pd.to_datetime(dfD['Time'].astype(str).str.strip(),
                               format='%d-%m-%Y %H:%M:%S', errors='coerce')

# ----------------------------
# Numeric columns (Surface)
# ----------------------------
pressure_S          = get_or_nan(dfS, 'Treating Pressure')
flowrate_S          = get_or_nan(dfS, 'Flow Rate') * 0.06 # l/min -> m³/hour
density_S           = get_or_nan(dfS, 'Density')
volume_S            = get_or_nan(dfS, 'Volume')
tot_volume_nrd_S    = get_or_nan(dfS, 'TOT vol nrd')
volume_nrd_S        = get_or_nan(dfS, 'NRD VOLUME')
return_volume_S     = get_or_nan(dfS, 'RETURN VOL')
totbalance_volume_S = get_or_nan(dfS, 'total bla')  # may not exist in some files
RDA_pressure_S      = get_or_nan(dfS, 'RDA2 IN1')

# ----------------------------
# Numeric columns (Downhole)
# ----------------------------
delta_time_D  = to_num(dfD['Delta Time'])
pressure_D    = to_num(dfD['Pressure']) * 0.0689476  # psi -> bar
temperature_D = to_num(dfD['Temperature'])

# ----------------------------
# Time windows
# ----------------------------
startS = pd.to_datetime("2023-12-10 15:15:00")
endS   = pd.to_datetime("2023-12-10 17:30:00")

startD = pd.to_datetime("2023-12-10 14:15:00")
endD   = pd.to_datetime("2023-12-10 16:30:00")

# ----------------------------
# Masks (NaN-safe)
# ----------------------------
maskS = (
    pressure_S.notna() &
    time_surface.notna() &
    (RDA_pressure_S.fillna(np.inf) <= 2000) &
    (time_surface >= startS) & (time_surface <= endS)
)

time_S               = time_surface[maskS]
pressure_S           = pressure_S[maskS]
flowrate_S           = flowrate_S[maskS]
density_S            = density_S[maskS]
volume_S             = volume_S[maskS]
tot_volume_nrd_S     = tot_volume_nrd_S[maskS]
volume_nrd_S         = volume_nrd_S[maskS]
return_volume_S      = return_volume_S[maskS]
totbalance_volume_S  = totbalance_volume_S[maskS]
RDA_pressure_S       = RDA_pressure_S[maskS]

print('surface time min/max:', (time_surface.min(), time_surface.max()))

maskD = (
    time_downhole.notna() &
    pressure_D.notna() &
    temperature_D.notna() &
    (time_downhole >= startD) & (time_downhole <= endD)
)

time_D        = time_downhole[maskD]
delta_time_D  = delta_time_D[maskD]
pressure_D    = pressure_D[maskD]
temperature_D = temperature_D[maskD]

print('downhole time min/max:', (time_downhole.min(), time_downhole.max()))
print('pressure (surface) head:', pressure_S.head(3).to_list())
print('pressure (downhole) head:', pressure_D.head(3).to_list())

# ----------------------------
# Surface panels
# ----------------------------
fig1, axs1 = plotting.plot_surface_panels(
    time_S, pressure_S, flowrate_S, density_S, volume_S,
    tot_volume_nrd_S, return_volume_S, totbalance_volume_S, RDA_pressure_S,
    figsize=(22, 8), markersize=0.2, linewidth=1.0, tick_labelsize=7, title_labelsize=6
)

# ----------------------------
# Triple-axis (surface subset)
# ----------------------------
fig2, (ax1, ax2, ax3) = plotting.plot_triple_axis(
    time_S, pressure_S, flowrate_S, return_volume_S,
    start=startS, end=endS, figsize=(16, 6)
)

# ----------------------------
# Downhole P/T plot
# ----------------------------
fig3, (ax4, ax5) = plotting.plot_downhole_pt(time_D, pressure_D, temperature_D, figsize=(16, 6))

# ----------------------------
# Depths & lag estimation
# ----------------------------
# Example MD/TVD arrays; replace with real values
MD = np.array([1364, 1383, 1383.3], dtype=float)
TVD = np.array([1363.18, np.nan, 1382.31], dtype=float)  # NaN = missing value
TVD_fracture_m = 1866.50
gauge_index    = 1  # index of the downhole gauge in the MD/TVD arrays

# NOTE: we use the (MD, TVD, gauge_index, TVD_fracture_m) signature per the fixed helper
TVD_interp, TVD_gauge_m, delta_tvd_m = well_corrections.estimate_lag(MD, TVD, gauge_index, TVD_fracture_m)

# Estimate delay (surface vs downhole)
try:
    lag_s, grid_step = time_difference.estimate_delay_seconds_robust(
        time_surface, pressure_S,
        time_downhole, pressure_D,
        max_lag_s=4*3600,    # 4 hours
        detrend_window_s=120
    )
except Exception as e:
    print("Delay estimation failed; defaulting lag_s=0. Reason:", e)
    lag_s = 0.0
print(f"Estimated delay: {float(lag_s)/3600:.2f} h")
# To align SURFACE to DOWNHOLE we shift SURFACE by **-lag_s**.

# ----------------------------
# Hydrostatic corrections to fracture depth
# ----------------------------
# Surface pressure corrected to fracture (if surface sensor is at TVD≈0, ΔTVD≈TVD_fracture_m)
p_surface_corr, _ = well_corrections.hydrostatic_correct_to_fracture(
    p_gauge=pressure_S,      # SURFACE treating pressure [bar]
    time_gauge=time_S,
    rho_surface=density_S,   # surface density series
    time_surface=time_S,
    delta_tvd_m=TVD_fracture_m,  # surface(0) -> fracture
    out_units='bar',
    lag_s=None
)

# Downhole gauge corrected to fracture
p_downhole_corr, _ = well_corrections.hydrostatic_correct_to_fracture(
    p_gauge=pressure_D,      # DOWNHOLE gauge [bar]
    time_gauge=time_D,
    rho_surface=density_S,   # use surface density aligned to DH via lag_s
    time_surface=time_S,
    delta_tvd_m=delta_tvd_m, # TVD_fracture - TVD_gauge
    out_units='bar',
    lag_s=lag_s
)

# ----------------------------
# Build aligned timelines (DOWNHOLE clock)
# ----------------------------
surface_dt_orig  = pd.to_datetime(time_S)
downhole_dt_orig = pd.to_datetime(time_D)

# Shift SURFACE timestamps by **-lag_s** to align onto DOWNHOLE clock
surface_dt_aligned_to_dh = surface_dt_orig - pd.to_timedelta(float(lag_s), unit='s')

y_surface  = pd.Series(p_surface_corr, copy=False).astype(float).to_numpy()
y_downhole = pd.Series(p_downhole_corr, copy=False).astype(float).to_numpy()

# Plot alignment (datetime)
figA, axA = plotting.plot_alignment(surface_dt_aligned_to_dh, y_surface, downhole_dt_orig, y_downhole, figsize=(16, 6))

# ----------------------------
# Closure analysis (DOWNHOLE time basis)
# ----------------------------
# Auto shut-in detection on SURFACE clock
t_shut_in_auto_surface = closure_analysis.find_shut_in_time_from_flow(
    time_S, flowrate_S, threshold=0.1, min_hold_s=10
)
print("Auto-detected shut-in (surface clock):", t_shut_in_auto_surface)

# Choose shut-in on SURFACE clock
t_shut_in_surface = t_shut_in_auto_surface if t_shut_in_auto_surface is not None else pd.to_datetime(time_S.iloc[-1])

# Convert shut-in to DOWNHOLE clock by shifting **-lag_s**
t_shut_in_downhole = t_shut_in_surface - pd.to_timedelta(float(lag_s), unit='s')
print("Shut-in used for DH analysis (downhole clock):", t_shut_in_downhole)

# Build downhole falloff series using DOWNHOLE times + chosen shut-in (on DOWNHOLE clock)
ts_dh, p_dh = closure_analysis.build_shut_in_series(time_D, p_downhole_corr, t_shut_in_downhole)
print(f"Falloff samples: {len(ts_dh)}; duration = {ts_dh[-1]:.1f} s")

# Square-root-of-time diagnostic
x_sqrt, p_srt, dpdx = closure_analysis.derivative_vs_sqrt_time(ts_dh, p_dh, max_t_s=180)
figSRT, axesSRT = plotting.plot_srt(x_sqrt, p_srt, dpdx)
for ax in axesSRT:
    ax.set_xlim(0, np.sqrt(180.0))

# Suggest closure candidate (first derivative minimum after ~1–2 min)
i_cl = closure_analysis.suggest_closure_from_srt(x_sqrt, p_srt, dpdx, min_t_s=0, guard_s=2, max_t_s=180)
if i_cl is not None:
    for a in axesSRT:
        a.axvline(x_sqrt[i_cl], ls='--')
    p_closure = p_srt[i_cl]
    print(f"Suggested closure (SRT): t = {x_sqrt[i_cl]**2:.1f} s,  P ≈ {p_closure:.2f} bar")

# Bourdet diagnostic (within 0–180 s)
t_log, dP_dlogt = closure_analysis.bourdet_derivative(ts_dh, p_dh, smooth_win=None, max_t_s=180)
i_brd = closure_analysis.suggest_closure_from_bourdet(t_log, dP_dlogt,
                                                      min_t_s=0, max_t_s=180)

figBRD, axBRD = plotting.plot_bourdet(t_log, dP_dlogt, x_sqrt=None, i_cl=None)
axBRD.set_xlim(0, 180)
if i_brd is not None:
    axBRD.axvline(t_log[i_brd], ls='--', color='k')
    axBRD.set_title(f"Bourdet – min at ~{t_log[i_brd]:.0f}s, P≈{p_dh[i_brd]:.2f} bar")

# ----------------------------
# MULTI-CYCLE: detect all pump-in → shut-in cycles (surface clock)
# and cut each falloff at (pump restart | flow-back start | end of data)
# ----------------------------
cycles = closure_analysis.detect_pump_cycles(
    time_S=time_S,
    q_m3h=flowrate_S,
    return_vol_S=return_volume_S,   # pass None if not available
    q_low=0.1, q_high=0.3,
    min_hold_s=30,
    min_gap_s=60,
    flowback_rate_thresh=0.2
)
print(f"Detected {len(cycles)} cycles.")

df_cycles = closure_analysis.analyze_all_shutins(
    cycles,
    time_S=time_S, flowrate_S=flowrate_S, return_volume_S=return_volume_S,
    time_D=time_D, p_downhole_corr=p_downhole_corr,
    lag_s=lag_s,
    min_falloff_s=120,
    min_t_s_for_pick=90,
    max_analysis_s=180,
    pick_from="bourdet"   # <--- use Bourdet min
)

# ----------------------------
# PER-CYCLE diagnostics & plots (for every detected cycle)
# ----------------------------
MIN_T_S_FOR_PICK = 90  # ignore the first 90 s
CAP_S = 180            # analyze/plot max 3 minutes

if len(cycles) == 0:
    print("No cycles detected — nothing to analyze.")
else:
    print("\nPer-cycle closure analysis (Bourdet min within 0–180 s):")
    for i, cyc in enumerate(cycles, 1):
        # Convert cycle window to DOWNHOLE clock
        t_shut_D = cyc['t_shut_in_surface'] - pd.to_timedelta(float(lag_s), unit='s')
        t_end_D  = cyc['t_end_surface']      - pd.to_timedelta(float(lag_s), unit='s')

        # Build + cap falloff series
        ts_dh, p_dh = closure_analysis.build_shut_in_series(time_D, p_downhole_corr, t_shut_D)
        t_end_rel   = (pd.to_datetime(t_end_D) - pd.to_datetime(t_shut_D)).total_seconds()
        t_hard_cap  = min(float(t_end_rel), float(CAP_S))
        keep        = (ts_dh <= t_hard_cap)
        ts_dh, p_dh = ts_dh[keep], p_dh[keep]

        # Require some usable falloff
        if len(ts_dh) < 3 or (ts_dh[-1] if len(ts_dh) else 0) < 120:
            print(f"  Cycle {i}: skipped (too short falloff: {ts_dh[-1] if len(ts_dh) else 0:.0f}s).")
            continue

        # --- Bourdet derivative pick (global minimum) ---
        t_log, dP_dlogt = closure_analysis.bourdet_derivative(ts_dh, p_dh, smooth_win=None, max_t_s=CAP_S)
        i_brd = closure_analysis.suggest_closure_from_bourdet(
            t_log, dP_dlogt, min_t_s=MIN_T_S_FOR_PICK, max_t_s=CAP_S
        )

        # Optional: √t view (for QC)
        x_sqrt, p_srt, dpdx = closure_analysis.derivative_vs_sqrt_time(ts_dh, p_dh, max_t_s=CAP_S)
        # If you also want to mark the √t minimum:
        i_srt = closure_analysis.suggest_closure_from_srt(x_sqrt, p_srt, dpdx,
                                                          min_t_s=MIN_T_S_FOR_PICK, max_t_s=CAP_S)

        # --- Plot SRT with min marker ---
        figSRT, axesSRT = plotting.plot_srt(x_sqrt, p_srt, dpdx)
        for ax in axesSRT:
            ax.set_xlim(0, np.sqrt(CAP_S))
        if i_srt is not None:
            for ax in axesSRT:
                ax.axvline(x_sqrt[i_srt], ls='--', color='k', alpha=0.8)

        # --- Plot Bourdet with min marker (authoritative pick) ---
        figBRD, axBRD = plotting.plot_bourdet(t_log, dP_dlogt, x_sqrt=None, i_cl=None)
        axBRD.set_xlim(0, CAP_S)
        if i_brd is not None:
            axBRD.axvline(t_log[i_brd], ls='--', color='k', alpha=0.9)
            axBRD.set_title(f"Cycle {i} — Bourdet min at ~{t_log[i_brd]:.0f}s, P≈{p_dh[i_brd]:.2f} bar")

        # Print the result
        if i_brd is not None:
            print(f"  Cycle {i}: FCP ≈ {p_dh[i_brd]:.2f} bar at t ≈ {t_log[i_brd]:.0f} s "
                  f"(ended by {cyc['ended_by']}, window {t_hard_cap:.0f}s)")
        else:
            print(f"  Cycle {i}: no Bourdet minimum found in [{MIN_T_S_FOR_PICK}, {CAP_S}] s.")



# --- WELLBORE STORAGE: estimate C_well from pre-breakdown PV and compute q_perf ---

# 1) Choose a pre-breakdown window on the SURFACE clock (edit these to your test)
pre_start = pd.to_datetime("2023-12-10 15:40:00")
pre_end   = pd.to_datetime("2023-12-10 16:00:00")
mask_pre  = (time_S >= pre_start) & (time_S <= pre_end)

# 2) Estimate C_well = dV/dp [m^3/bar] from PV slope on that linear segment
C_well, stats = well_corrections.estimate_wellbore_compliance_from_pv(
    time_S, pressure_S, volume_S, mask=mask_pre
)
print(f"C_well ≈ {C_well:.4f} m³/bar  (n={stats['n']}, R²={stats['r2']:.3f}, "
      f"p∈[{stats['p_range'][0]:.1f},{stats['p_range'][1]:.1f}] bar)")

# 3) Compute q@perfs from surface flow and dp/dt: q_perf = q_pump - C_well * dp/dt
q_perf_S, diag = well_corrections.flow_at_perfs_from_surface(
    q_pump=flowrate_S,          # surface flow (usually m³/h)
    time_q=time_S,              # same clock as pump logger
    pressure_bar=pressure_S,    # treating pressure (bar)
    time_p=time_S,              # same clock as pressure
    C_well_m3_per_bar=C_well,
    out_units='m3/h',
    smooth_dpdt_s=5             # small smoothing for dp/dt
)
print("q_perf head [m³/h]:", q_perf_S.head(3).to_list())

# 4) (Optional) Plot q_pump vs q_perf and C_well*dp/dt

fig_q, axq = plt.subplots(figsize=(14, 5))
axq.plot(time_S, flowrate_S, label='q_pump (surface) [m³/h]', lw=1)
axq.plot(time_S, q_perf_S,   label='q_perf @ perfs [m³/h]', lw=1)
axq.set_ylabel('Flowrate [m³/h]')
axq.set_title('Wellbore storage correction: q_perf = q_pump - C_well dp/dt')
axq.grid(True, alpha=0.4)
axq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axq.legend(loc='upper left')

axq2 = axq.twinx()
axq2.plot(time_S, (diag['dpdt_bar_per_s_on_q']*C_well*3600.0),  # (m³/s)*3600 = m³/h
          ls='--', lw=1, label='C_well * dp/dt [m³/h]')
axq2.set_ylabel('C_well·dp/dt [m³/h]')
l1, lab1 = axq.get_legend_handles_labels()
l2, lab2 = axq2.get_legend_handles_labels()
axq.legend(l1+l2, lab1+lab2, loc='upper center', fontsize=8)

fig_q.autofmt_xdate()
fig_q.tight_layout()

plt.show()

