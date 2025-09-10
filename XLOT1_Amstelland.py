import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# Custom/local modules (assumed available on PYTHONPATH)
import time_difference
import well_corrections
import closure_analysis


# ----------------------------
# Helpers
# ----------------------------
def to_num(series):
    """
    Normalize numeric text:
    - strip whitespace
    - remove NBSPs
    - convert decimal commas to dots
    - coerce invalids to NaN
    Returns a pandas Series of dtype float.
    """
    s = pd.Series(series, copy=False).astype(str).str.strip()
    s = s.str.replace('\u00A0', '', regex=False)  # NBSP
    s = s.str.replace(',', '.', regex=False)      # decimal comma -> dot
    s = s.replace({'': None})
    return pd.to_numeric(s, errors='coerce')


# ----------------------------
# Load data
# ----------------------------
df_AMS_XLOT_1_S = pd.read_csv(
    r'C:\Users\36832-544\OneDrive - EBN BV\Documents\python_scripts\Data\Amstelland\NLOG_GS_PUB_AMS-01_XLOT 1 - Main Claystone_Cementing Unit_Data.txt',
    sep='\t',
    comment='_',
    skiprows=0,
    engine='python',
    on_bad_lines='skip'
)

df_AMS_XLOT_1_D = pd.read_csv(
    r'C:\Users\36832-544\OneDrive - EBN BV\Documents\python_scripts\Data\Amstelland\NLOG_GS_PUB_AMS-01_XLOT 1 - Main Claystone_Downhole Gauge_Data.TXT',
    sep='\t',
    comment='_',
    skiprows=0,
    engine='python',
    on_bad_lines='skip'
)

# Ensure expected columns exist for downhole file
df_AMS_XLOT_1_D.columns = [
    'Time',
    'Delta Time',
    'Pressure',
    'Temperature',
    'None',
    'None2'
]

# Create numpy arrays for all columns in the dataframes
arrays_dict1 = {col: np.array(df_AMS_XLOT_1_S[col][1:]) for col in df_AMS_XLOT_1_S.columns}
arrays_dict2 = {col: np.array(df_AMS_XLOT_1_D[col].iloc[4:]) for col in df_AMS_XLOT_1_D.columns[:-2]}

# ----------------------------
# Parse time columns
# ----------------------------
# Surface time: format like %m:%d:%Y:%H:%M:%S
time_array1 = pd.Series(
    pd.to_datetime(arrays_dict1['Time'], format='%m:%d:%Y:%H:%M:%S', errors='coerce'),
    name="Time"
)

# Downhole time: format like '09-12-2023 21:24:09'
time_array2 = pd.Series(
    pd.to_datetime(pd.Series(arrays_dict2['Time']).astype(str).str.strip(),
                   format='%d-%m-%Y %H:%M:%S', errors='coerce'),
    name="Time"
)

print(time_array1)
print(time_array2)

# ----------------------------
# Convert numeric columns (Surface)
# ----------------------------
pressure_array          = to_num(arrays_dict1.get('Treating Pressure'))
flowrate_array          = to_num(arrays_dict1.get('Flow Rate'))
density_array           = to_num(arrays_dict1.get('Density'))
volume_array            = to_num(arrays_dict1.get('Volume'))
tot_volume_nrd_array    = to_num(arrays_dict1.get('TOT vol nrd'))
volume_nrd_array        = to_num(arrays_dict1.get('NRD VOLUME'))
return_volume_array     = to_num(arrays_dict1.get('RETURN VOL'))
totbalance_volume_array = to_num(arrays_dict1.get('total bla'))
RDA_pressure_array      = to_num(arrays_dict1.get('RDA2 IN1'))

# ----------------------------
# Convert numeric columns (Downhole)
# ----------------------------
delta_time_array2 = to_num(arrays_dict2['Delta Time'])
pressure_array2   = to_num(arrays_dict2['Pressure']) * 0.0689476  # psi -> bar
temperature_array2 = to_num(arrays_dict2['Temperature'])

# ----------------------------
# Time windows
# ----------------------------
start1 = pd.to_datetime("2023-12-10 15:15:00")
end1   = pd.to_datetime("2023-12-10 17:30:00")

start2 = pd.to_datetime("2023-12-10 14:15:00")
end2   = pd.to_datetime("2023-12-10 16:30:00")

# ----------------------------
# Masks (NaN-safe)
# ----------------------------
mask1 = (
    pressure_array.notna() &
    time_array1.notna() &
    (RDA_pressure_array.fillna(np.inf) <= 2000) &
    (time_array1 >= start1) &
    (time_array1 <= end1)
)

time_array               = time_array1[mask1]
pressure_array           = pressure_array[mask1]
flowrate_array           = flowrate_array[mask1]
density_array            = density_array[mask1]
volume_array             = volume_array[mask1]
tot_volume_nrd_array     = tot_volume_nrd_array[mask1]
volume_nrd_array         = volume_nrd_array[mask1]
return_volume_array      = return_volume_array[mask1]
totbalance_volume_array  = totbalance_volume_array[mask1]
RDA_pressure_array       = RDA_pressure_array[mask1]

print('time_array1 min, time_array1 max: ', (time_array1.min(), time_array1.max()))

mask2 = (
    time_array2.notna() &
    pressure_array2.notna() &
    temperature_array2.notna() &
    (time_array2 >= start2) &
    (time_array2 <= end2)
)

time_array2        = time_array2[mask2]
delta_time_array2  = delta_time_array2[mask2]
pressure_array2    = pressure_array2[mask2]
temperature_array2 = temperature_array2[mask2]

print('time_array2 min, time_array2 max: ', (time_array2.min(), time_array2.max()))

print('pressure_array (surface): ', pressure_array)
print('pressure_array2 (downhole): ', pressure_array2)

def function_plot1(
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
    markersize=0.1,
    linewidth=0.5,
    tick_labelsize=7,
    title_labelsize=6,
):
    """Create 2x4 surface panels vs time."""

    # Each tuple: (y_array, y_label, panel_title, color)
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
        ax.plot(time_array, y, color=color, marker='o', markersize=markersize, linestyle='-', linewidth=linewidth)
        ax.set_xlabel('Time', **font_small)
        ax.set_ylabel(ylabel, **font_small)
        ax.set_title(ptitle, **font_small)
        ax.tick_params(axis='both', labelsize=tick_labelsize)
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    return fig, axs

function_plot1(
    time_array,
    pressure_array,
    flowrate_array,
    density_array,
    volume_array,
    tot_volume_nrd_array,
    return_volume_array,
    totbalance_volume_array,
    RDA_pressure_array,
    figsize=(22, 8),
    markersize=0.2,
    linewidth=1.0,
    tick_labelsize=7,
    title_labelsize=6,
)

# === Format x-axis to show only HH:MM on all subplots ===
for ax in axs.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # Optional: control tick spacing, e.g. every 5 minutes
    # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

fig.autofmt_xdate()  # nicer label rotation/spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

# ---------- Figure 2: combined triple-axis ----------
font_small = {'fontsize': 8}
fig2, ax1 = plt.subplots(figsize=(16, 6))

# Left y-axis: Pressure
ax1.plot(
    time_array,
    pressure_array,
    color='royalblue',
    linestyle='-',
    linewidth=1,
    label='Pressure [bar]'
)
ax1.set_xlabel('Time', **font_small)
ax1.set_ylabel('Pressure [bar]', **font_small)
ax1.set_title('Pressure, Flowrate & Return Volume vs Time', **font_small)
ax1.tick_params(axis='both', labelsize=7)
ax1.set_xlim(left=start1, right=end1)
# ax1.set_ylim(top=240)
ax1.grid()

# First right y-axis: Flowrate
ax2 = ax1.twinx()
ax2.plot(
    time_array,
    flowrate_array,
    color='forestgreen',
    linestyle='-',
    linewidth=1,
    label='Flowrate [m³/h]'
)
ax2.set_ylabel('Flowrate [m³/h]', **font_small)
ax2.tick_params(axis='y', labelsize=7)
# ax2.set_ylim(top=400)

# Second right y-axis: Return Volume
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
ax3.plot(
    time_array,
    return_volume_array,
    color='darkorange',
    linestyle='--',
    linewidth=1,
    label='Return Volume [m³]'
)
ax3.set_ylabel('Return Volume [m³]', **font_small)
ax3.tick_params(axis='y', labelsize=7)

# === Format x-axis to show only HH:MM (twins share x-axis with ax1) ===
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# Optional: set tick spacing
# ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
ax1.legend(
    lines_1 + lines_2 + lines_3,
    labels_1 + labels_2 + labels_3,
    fontsize=6,
    loc='upper center'
)

fig2.autofmt_xdate()
plt.tight_layout()

# ----------------------------
# Figure 3: Pressure & Temperature vs Time (downhole, unchanged)
# ----------------------------
fig3, ax4 = plt.subplots(figsize=(16, 6))
ax4.plot(time_array2, pressure_array2, color='royalblue', linestyle='-', linewidth=1, label='Pressure [bar]')
ax4.set_xlabel('Time', fontsize=8)
ax4.set_ylabel('Pressure [bar]', fontsize=8)
ax4.set_title('Pressure and Temperature vs Time', fontsize=8)
ax4.tick_params(axis='both', labelsize=7)
ax4.grid()
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax5 = ax4.twinx()
ax5.plot(time_array2, temperature_array2, color='forestgreen', linestyle='-', linewidth=1, label='Temperature [°C]')
ax5.set_ylabel('Temperature [°C]', fontsize=8)
ax5.tick_params(axis='y', labelsize=7)

lines4, labels4 = ax4.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()
ax4.legend(lines4 + lines5, labels4 + labels5, fontsize=6, loc='upper left')

fig3.autofmt_xdate()
plt.tight_layout()

# ----------------------------
# Depths and lag estimation
# ----------------------------
# Example MD/TVD arrays; fill with your real values
MD = np.array([1364, 1383, 1383.3])
TVD = np.array([1363.18, np.nan, 1382.31])  # NaN = missing value
TVD_fracture_m=1866.50
gauge_index=1  # index of the downhole gauge in the MD/TVD arrays

TVD_interp, TVD_gauge_m, delta_tvd_m = well_corrections.estimate_lag(MD, TVD, gauge_index, TVD_fracture_m)

# Estimate delay (surface vs downhole)
lag_s, grid_step = time_difference.estimate_delay_seconds_robust(
    time_array1, pressure_array,
    time_array2, pressure_array2,
    max_lag_s=4*3600,
    detrend_window_s=120
)
print(f"Estimated delay: {lag_s/3600:.2f} h")
# NOTE (fixed): to align SURFACE to DOWNHOLE we now shift SURFACE by **-lag_s**.

# ----------------------------
# Hydrostatic corrections to fracture depth (on original time bases)
# ----------------------------
p_surface_corrected, _ = well_corrections.hydrostatic_correct_to_fracture(
    p_gauge=pressure_array,      # SURFACE pressure (bar)
    time_gauge=time_array,
    rho_surface=density_array,   # surface density series (kg/m^3)
    time_surface=time_array,
    delta_tvd_m=TVD_fracture_m,  # surface -> fracture vertical distance (m)
    out_units='bar',
    lag_s=None                   # keep physical correction on original time basis
)

p_downhole_corrected, _ = well_corrections.hydrostatic_correct_to_fracture(
    p_gauge=pressure_array2,     # DOWNHOLE gauge pressure (bar)
    time_gauge=time_array2,
    rho_surface=density_array,   # surface density series
    time_surface=time_array,
    delta_tvd_m=delta_tvd_m,     # gauge -> fracture vertical distance (m)
    out_units='bar',
    lag_s=lag_s                  # align density sampling to downhole timestamps
)

# Ensure lag_s defined
lag_s = 0 if ('lag_s' not in globals() or lag_s is None) else float(lag_s)

# ----------------------------
# ALIGNMENT: Surface → Downhole time basis (FIXED DIRECTION)
# ----------------------------
surface_dt_orig  = pd.to_datetime(time_array)
downhole_dt_orig = pd.to_datetime(time_array2)

# Shift SURFACE timestamps by **-lag_s** to land on the DOWNHOLE clock
surface_dt_aligned_to_dh = surface_dt_orig - pd.to_timedelta(lag_s, unit='s')

# Build relative-seconds axes using DOWNHOLE as the reference
ts_downhole_secs_ref = well_corrections.series_to_seconds(time_array2)               # unchanged
ts_surface_secs_algn = well_corrections.series_to_seconds(time_array) - lag_s        # shifted onto downhole frame

y_surface  = pd.Series(p_surface_corrected, copy=False).astype(float).to_numpy()
y_downhole = pd.Series(p_downhole_corrected, copy=False).astype(float).to_numpy()
assert len(ts_surface_secs_algn) == len(y_surface)
assert len(ts_downhole_secs_ref) == len(y_downhole)

# ----------------------------
# Figure A: relative seconds (DOWNHOLE reference)
# ----------------------------
#fig_rel, ax_rel = plt.subplots(figsize=(16, 6))
#ax_rel.plot(ts_downhole_secs_ref, y_downhole, label='Downhole → fracture [bar]', linewidth=1)
#ax_rel.plot(ts_surface_secs_algn, y_surface,  label='Surface (shifted to downhole) → fracture [bar]', linewidth=1, alpha=0.9)
#ax_rel.set_xlabel('Time since start [s] (downhole reference)', fontsize=8)
#ax_rel.set_ylabel('Pressure [bar]', fontsize=8)
#ax_rel.set_title('Surface shifted to Downhole timeline (both corrected to fracture)', fontsize=9)
#ax_rel.tick_params(axis='both', labelsize=7)
#ax_rel.grid(True, alpha=0.4)
#ax_rel.legend()

# ----------------------------
# Figure B: pretty datetime axis (DOWNHOLE clock)
# ----------------------------
# Downhole stays on its original clock
ts_downhole_dt = downhole_dt_orig
# Surface is shifted by **-lag_s**
ts_surface_dt_algn = surface_dt_aligned_to_dh

fig_dt, ax_dt = plt.subplots(figsize=(16, 6))
ax_dt.plot(ts_surface_dt_algn, y_surface,  label='Surface (shifted to downhole) → fracture [bar]', linewidth=1, alpha=0.9)
ax_dt.plot(ts_downhole_dt, y_downhole, label='Downhole → fracture [bar]', linewidth=1)
ax_dt.set_xlabel('Time (HH:MM) — downhole clock', fontsize=8)
ax_dt.set_ylabel('Pressure [bar]', fontsize=8)
ax_dt.set_title('Surface shifted to Downhole timeline (both corrected to fracture)', fontsize=9)
ax_dt.tick_params(axis='both', labelsize=7)
ax_dt.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_dt.grid(True, alpha=0.4)
ax_dt.legend()




# ----------------------------
# Figure 3: Pressure & Temperature vs Time (downhole, unchanged)
# ----------------------------
fig3, ax4 = plt.subplots(figsize=(16, 6))
ax4.plot(time_array2, pressure_array2, color='royalblue', linestyle='-', linewidth=1, label='Pressure [bar]')
ax4.set_xlabel('Time', fontsize=8)
ax4.set_ylabel('Pressure [bar]', fontsize=8)
ax4.set_title('Pressure and Temperature vs Time', fontsize=8)
ax4.tick_params(axis='both', labelsize=7)
ax4.grid()
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax5 = ax4.twinx()
ax5.plot(time_array2, temperature_array2, color='forestgreen', linestyle='-', linewidth=1, label='Temperature [°C]')
ax5.set_ylabel('Temperature [°C]', fontsize=8)
ax5.tick_params(axis='y', labelsize=7)

lines4, labels4 = ax4.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()
ax4.legend(lines4 + lines5, labels4 + labels5, fontsize=6, loc='upper left')

fig3.autofmt_xdate()
plt.tight_layout()

# ----------------------------
# Closure analysis (DOWNHOLE time basis) — FIXED DIRECTION
# ----------------------------
# Auto shut-in detection from SURFACE flow (on surface clock)...
t_shut_in_auto_surface = closure_analysis.find_shut_in_time_from_flow(
    time_array, flowrate_array, threshold=0.1, min_hold_s=30
)
print("Auto-detected shut-in (surface clock):", t_shut_in_auto_surface)

# Choose shut-in on surface clock (auto or last surface time)
t_shut_in_surface = t_shut_in_auto_surface if t_shut_in_auto_surface is not None else pd.to_datetime(time_array.iloc[-1])

# Convert shut-in to DOWNHOLE clock by shifting **-lag_s**
t_shut_in_downhole = t_shut_in_surface - pd.to_timedelta(lag_s, unit='s')
print("Shut-in used for DH analysis (downhole clock):", t_shut_in_downhole)

# Build downhole falloff series using DOWNHOLE times + chosen shut-in on DOWNHOLE clock
ts_dh, p_dh = closure_analysis.build_shut_in_series(time_array2, p_downhole_corrected, t_shut_in_downhole)
print(f"Falloff samples: {len(ts_dh)}; duration = {ts_dh[-1]:.1f} s")

# Square-root-of-time diagnostic
x_sqrt, p_srt, dpdx = closure_analysis.derivative_vs_sqrt_time(ts_dh, p_dh)

fig_srt, ax_srt = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax_srt[0].plot(x_sqrt, p_srt, lw=1)
ax_srt[0].set_ylabel("P @ fracture [bar]")
ax_srt[0].set_title("Closure analysis — Square-root-of-time (downhole clock)")

ax_srt[1].plot(x_sqrt, dpdx, lw=1)
ax_srt[1].set_xlabel("sqrt(time since shut-in) [s^0.5]")
ax_srt[1].set_ylabel("dP/d√t [bar / s^0.5]")
ax_srt[1].grid(True, alpha=0.4)

# Suggest closure candidate (first derivative minimum after ~1–2 min)
i_cl = closure_analysis.suggest_closure_from_srt(x_sqrt, p_srt, dpdx, min_t_s=90, guard_s=3)
if i_cl is not None:
    for a in ax_srt:
        a.axvline(x_sqrt[i_cl], ls='--')
    p_closure = p_srt[i_cl]
    print(f"Suggested closure (SRT): t = {x_sqrt[i_cl]**2:.1f} s,  P ≈ {p_closure:.2f} bar")

plt.tight_layout()

# Bourdet (semilog) derivative vs log time
t_log, dP_dlogt = closure_analysis.bourdet_derivative(ts_dh, p_dh, smooth_win=None)

fig_brd, ax_brd = plt.subplots(figsize=(12, 5))
ax_brd.semilogx(t_log, dP_dlogt, lw=1)
ax_brd.set_xlabel("time since shut-in [s] (log) — downhole clock")
ax_brd.set_ylabel("dP/d(ln t) [bar]")
ax_brd.set_title("Closure analysis — Bourdet derivative (downhole clock)")
ax_brd.grid(True, which='both', alpha=0.4)

# Mark SRT suggestion on derivative plot
if i_cl is not None:
    ax_brd.axvline(x_sqrt[i_cl]**2, ls='--')

plt.tight_layout()
plt.show()
