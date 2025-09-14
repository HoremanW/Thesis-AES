from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# Local modules
import time_difference
import well_corrections
import closure_analysis
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
flowrate_S          = get_or_nan(dfS, 'Flow Rate') * 0.06  # l/min -> m³/h
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

# Estimate delay (surface vs downhole) — IMPORTANT: use masked series (time_S/time_D)
try:
    lag_s, grid_step = time_difference.estimate_delay_seconds_robust(
        time_S, pressure_S,          # masked/aligned series
        time_D, pressure_D,
        max_lag_s=4*3600,            # 4 hours
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
# Build aligned timelines (DOWNHOLE clock) – for visual QC
# ----------------------------
surface_dt_orig  = pd.to_datetime(time_S)
downhole_dt_orig = pd.to_datetime(time_D)
surface_dt_aligned_to_dh = surface_dt_orig - pd.to_timedelta(float(lag_s), unit='s')

y_surface  = pd.Series(p_surface_corr, copy=False).astype(float).to_numpy()
y_downhole = pd.Series(p_downhole_corr, copy=False).astype(float).to_numpy()

figA, axA = plotting.plot_alignment(surface_dt_aligned_to_dh, y_surface, downhole_dt_orig, y_downhole, figsize=(16, 6))

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

# ----------------------------
# Analyze each cycle on DOWNHOLE clock using corrected DH pressure
# ----------------------------
MIN_T_S_FOR_PICK = 1  # seconds to ignore early wellbore storage
df_cycles = closure_analysis.analyze_all_shutins(
    cycles,
    time_S=time_S, flowrate_S=flowrate_S, return_volume_S=return_volume_S,
    time_D=time_D, p_downhole_corr=p_downhole_corr,
    lag_s=lag_s,
    min_falloff_s=120,
    min_t_s_for_pick=MIN_T_S_FOR_PICK,  # <-- NEW: pick closure after 1 second
    max_analysis_s=180   # <-- NEW: only first 3 minutes after shut-in
)
print(df_cycles)

# ----------------------------
# OPTIONAL: visualize per-cycle falloff windows & picks
# ----------------------------
MIN_T_S_FOR_PICK = 1  # seconds to ignore early wellbore storage

for _, row in df_cycles.iterrows():
    if not row.get('usable', False):
        continue
    # reconstruct window on DH clock for plotting
    t_shut_D = row['t_shut_in_surface'] - pd.to_timedelta(float(lag_s), unit='s')
    t_end_D  = row['t_end_surface']      - pd.to_timedelta(float(lag_s), unit='s')

    ts_dh, p_dh = closure_analysis.build_shut_in_series(time_D, p_downhole_corr, t_shut_D)
    t_end_rel = (pd.to_datetime(t_end_D) - pd.to_datetime(t_shut_D)).total_seconds()
    t_hard_cap = min(float(t_end_rel), 180.0)
    keep = (ts_dh <= t_hard_cap)
    ts_dh, p_dh = ts_dh[keep], p_dh[keep]


    x_sqrt, p_srt, dpdx = closure_analysis.derivative_vs_sqrt_time(ts_dh, p_dh)
    i_cl = closure_analysis.suggest_closure_from_srt(
        x_sqrt, p_srt, dpdx,
        min_t_s=MIN_T_S_FOR_PICK,
        guard_n=5,
        max_t_s=180,
        prefer_global_min=True,
        dpdx_smooth_pts=5
    )

    figSRT, axesSRT = plotting.plot_srt(x_sqrt, p_srt, dpdx)
    if i_cl is not None:
        for a in axesSRT:
            a.axvline(x_sqrt[i_cl], ls='--', color='k')
    figSRT.suptitle(f"Cycle {int(row['cycle'])} – FCP ≈ {row['closure_pressure_bar']:.2f} bar")

# ----------------------------
# WELLBORE STORAGE example (unchanged)
# ----------------------------
pre_start = pd.to_datetime("2023-12-10 15:40:00")
pre_end   = pd.to_datetime("2023-12-10 16:00:00")
mask_pre  = (time_S >= pre_start) & (time_S <= pre_end)

C_well, stats = well_corrections.estimate_wellbore_compliance_from_pv(
    time_S, pressure_S, volume_S, mask=mask_pre
)
print(f"C_well ≈ {C_well:.4f} m³/bar  (n={stats['n']}, R²={stats['r2']:.3f}, "
      f"p∈[{stats['p_range'][0]:.1f},{stats['p_range'][1]:.1f}] bar)")

q_perf_S, diag = well_corrections.flow_at_perfs_from_surface(
    q_pump=flowrate_S,
    time_q=time_S,
    pressure_bar=pressure_S,
    time_p=time_S,
    C_well_m3_per_bar=C_well,
    out_units='m3/h',
    smooth_dpdt_s=5
)
print("q_perf head [m³/h]:", q_perf_S.head(3).to_list())

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

# --- Estimate pumping duration tp (from surface flow) ---
#def _first_pump_time(time_series, flow_series, threshold=0.1, hold_s=30):
#    t = pd.to_datetime(pd.Series(time_series))
#    q = pd.Series(flow_series, dtype=float)
    # first sustained rise above threshold
#    for i in range(len(q)):
#        if pd.notna(q.iloc[i]) and q.iloc[i] > threshold:
#            t0 = t.iloc[i]
            # hold check to skip noise
#            j = i
#            ok = True
#            while j < len(q) and (t.iloc[j] - t0).total_seconds() <= hold_s:
#                if pd.notna(q.iloc[j]) and q.iloc[j] <= threshold:
#                    ok = False; break
#                j += 1
#            if ok:
#                return t0
#    return None

#t_pump_start_surface = _first_pump_time(time_S, flowrate_S, threshold=0.1, hold_s=30)
#if t_pump_start_surface is None:
    # fallback: 30 minutes before shut-in
#    t_pump_start_surface = t_shut_in_surface - pd.to_timedelta(30, unit='m')

#tp_seconds = (t_shut_in_surface - t_pump_start_surface).total_seconds()
#print(f"Estimated pumping duration tp ≈ {tp_seconds/60:.1f} min")

# --- Barree Figures 1–4 (Normal Leakoff) on DOWNHOLE falloff series ---
#barree_figs = plotting.plot_barree_figs_normal_leakoff(
#    t_post_s=ts_dh,
#    p_post=p_dh,
#    tp_seconds=tp_seconds,
#    isip=None,                # default: first post-shut-in sample
#    closure_idx=None,         # let auto-closure from G semilog derivative pick it
#    auto_closure=True,
#    auto_fit_frac=0.25,
#    auto_tol=0.10,
#    pi_guess=None             # optional; derivative doesn't depend on this
#)

# optional: save
#for name, fig in barree_figs.items():
#    fig.suptitle(name)
#    fig.savefig(f"figures\{name}.png", dpi=200, bbox_inches='tight')


#plt.show()
