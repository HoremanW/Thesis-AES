import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates  # ← NEW: for time-only tick formatting

# Load the file
df_AMS_XLOT_1_S = pd.read_csv(
    r'C:\Users\36832-544\OneDrive - EBN BV\Documents\python_scripts\Data\Amstelland\NLOG_GS_PUB_AMS-01_XLOT 2 - Vlieland Claystone_Cementing Unit_Data.txt',
    sep='\t',
    comment='_',
    skiprows=0,
    engine='python',
    on_bad_lines='skip'
)

# Show first few rows
print(df_AMS_XLOT_1_S.head())

# Create numpy arrays for all columns in the dataframe
arrays_dict = {col: np.array(df_AMS_XLOT_1_S[col][1:]) for col in df_AMS_XLOT_1_S.columns}
print(arrays_dict.keys())

# Convert arrays to numeric/datetime
time_array = pd.to_datetime(arrays_dict['Time'], format='%d:%m:%Y:%H:%M:%S', errors='coerce')
pressure_array = pd.to_numeric(arrays_dict['Treating Pressure'], errors='coerce')
flowrate_array = pd.to_numeric(arrays_dict['Flow Rate'], errors='coerce')
density_array = pd.to_numeric(arrays_dict['Density'], errors='coerce')
volume_array = pd.to_numeric(arrays_dict['Volume'], errors='coerce')
cmt_return_press_array = pd.to_numeric(arrays_dict['CMT Return Press'], errors='coerce')
cmt_return_rate_array = pd.to_numeric(arrays_dict['CMT Return Rate'], errors='coerce')
cmt_return_rate_dens = pd.to_numeric(arrays_dict['CMT Return Dens'], errors='coerce')
return_volume_array = pd.to_numeric(arrays_dict['Return Volume'], errors='coerce')

print(time_array)

# Mask: use pd.notna for datetimes (instead of np.isnan)
mask = (
    ~np.isnan(pressure_array) &
    pd.notna(time_array)
)

time_array = time_array[mask]
start = pd.to_datetime(time_array.min().strftime('%Y-%m-%d') + ' 15:15:00')
end = pd.to_datetime(time_array.min().strftime('%Y-%m-%d') + ' 17:30:00')
pressure_array = pressure_array[mask]
flowrate_array = flowrate_array[mask]
density_array = density_array[mask]
volume_array = volume_array[mask]
cmt_return_press_array = cmt_return_press_array[mask]
cmt_return_rate_array = cmt_return_rate_array[mask]
cmt_return_rate_dens = cmt_return_rate_dens[mask]
return_volume_array = return_volume_array[mask]

# ---------- Figure 1: 2x4 subplots ----------
fig, axs = plt.subplots(2, 4, figsize=(22, 8))  # 2 rows, 4 columns
font_small = {'fontsize': 6}

# Treating Pressure vs Time
axs[0, 0].plot(time_array, pressure_array, color='green', marker='o', markersize=0.2, linestyle='-')
axs[0, 0].set_xlabel('Time', **font_small)
axs[0, 0].set_ylabel('Treating Pressure [bar]', **font_small)
axs[0, 0].set_title('Treating Pressure vs Time', **font_small)
axs[0, 0].tick_params(axis='both', labelsize=7)
axs[0, 0].grid()

# Volume vs Time
axs[0, 1].plot(time_array, volume_array, color='blue', marker='o', markersize=0.2, linestyle='-')
axs[0, 1].set_xlabel('Time', **font_small)
axs[0, 1].set_ylabel('Volume [m³]', **font_small)
axs[0, 1].set_title('Volume vs Time', **font_small)
axs[0, 1].tick_params(axis='both', labelsize=7)
axs[0, 1].grid()

# TOT Volume NRD vs Time
axs[0, 2].plot(time_array, tot_volume_nrd_array, color='purple', marker='o', markersize=0.2, linestyle='-')
axs[0, 2].set_xlabel('Time', **font_small)
axs[0, 2].set_ylabel('TOT Volume NRD [m³]', **font_small)
axs[0, 2].set_title('TOT Volume NRD vs Time', **font_small)
axs[0, 2].tick_params(axis='both', labelsize=7)
axs[0, 2].grid()

# Flowrate vs Time
axs[0, 3].plot(time_array, flowrate_array, color='red', marker='o', markersize=0.2, linestyle='-')
axs[0, 3].set_xlabel('Time', **font_small)
axs[0, 3].set_ylabel('Flowrate [m³/h]', **font_small)
axs[0, 3].set_title('Flowrate vs Time', **font_small)
axs[0, 3].tick_params(axis='both', labelsize=7)
axs[0, 3].grid()

# Density vs Time
axs[1, 0].plot(time_array, density_array, color='teal', marker='o', markersize=0.2, linestyle='-')
axs[1, 0].set_xlabel('Time', **font_small)
axs[1, 0].set_ylabel('Density [Kg/m³]', **font_small)
axs[1, 0].set_title('Density vs Time', **font_small)
axs[1, 0].tick_params(axis='both', labelsize=7)
axs[1, 0].grid()

# RETURN VOL vs Time
axs[1, 1].plot(time_array, return_volume_array, color='orange', marker='o', markersize=0.2, linestyle='-')
axs[1, 1].set_xlabel('Time', **font_small)
axs[1, 1].set_ylabel('RETURN VOL [m³]', **font_small)
axs[1, 1].set_title('RETURN VOL vs Time', **font_small)
axs[1, 1].tick_params(axis='both', labelsize=7)
axs[1, 1].grid()

# total bla vs Time
axs[1, 2].plot(time_array, totbalance_volume_array, color='purple', marker='o', markersize=0.2, linestyle='-')
axs[1, 2].set_xlabel('Time', **font_small)
axs[1, 2].set_ylabel('total bla [m³]', **font_small)
axs[1, 2].set_title('total bla vs Time', **font_small)
axs[1, 2].tick_params(axis='both', labelsize=7)
axs[1, 2].grid()

# RDA2 IN1 vs Time
axs[1, 3].plot(time_array, RDA_pressure_array, color='red', marker='o', markersize=0.2, linestyle='-')
axs[1, 3].set_xlabel('Time', **font_small)
axs[1, 3].set_ylabel('RDA2 IN1 [bar]', **font_small)
axs[1, 3].set_title('RDA2 IN1 vs Time', **font_small)
axs[1, 3].tick_params(axis='both', labelsize=7)
axs[1, 3].grid()

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
ax1.plot(time_array, pressure_array, color='royalblue', linestyle='-', linewidth=1, label='Pressure [bar]')
ax1.set_xlabel('Time', **font_small)
ax1.set_ylabel('Pressure [bar]', **font_small)
ax1.set_title('Pressure, Flowrate & Return Volume vs Time', **font_small)
ax1.tick_params(axis='both', labelsize=7)
ax1.set_xlim(left=start, right=end)
#ax1.set_ylim(top=240)   

ax1.grid()

# First right y-axis: Flowrate
ax2 = ax1.twinx()
ax2.plot(time_array, flowrate_array, color='forestgreen', linestyle='-', linewidth=1, label='Flowrate [m³/h]')
ax2.set_ylabel('Flowrate [m³/h]', **font_small)
ax2.tick_params(axis='y', labelsize=7)
#ax2.set_ylim(top=400)

# Second right y-axis: Return Volume
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
ax3.plot(time_array, return_volume_array, color='darkorange', linestyle='--', linewidth=1, label='Return Volume [m³]')
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
ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, fontsize=6, loc='upper center')

fig2.autofmt_xdate()
plt.tight_layout()





# Load the file
df_AMS_XLOT_1_D = pd.read_csv(
    r'C:\Users\36832-544\OneDrive - EBN BV\Documents\python_scripts\Data\Amstelland\NLOG_GS_PUB_AMS-01_XLOT 2 - Vlieland Claystone_Downhole Gauge_Data.TXT',
    sep='\t',
    comment='_',
    skiprows=0,
    engine='python',
    on_bad_lines='skip'
)

# Show first few rows
print(df_AMS_XLOT_1_D.head())

df_AMS_XLOT_1_D.columns = [
    'Time',
    'Delta Time',
    'Pressure',
    'Temperature',
    'None',
    'None2'
]

# Only take data from the 6th row onwards, excluding the last two columns
arrays_dict = {
    col: np.array(df_AMS_XLOT_1_D[col].iloc[4:])
    for col in df_AMS_XLOT_1_D.columns[:-2]
}

print(arrays_dict)
print(arrays_dict.keys())

# Convert arrays to numeric/datetime
# Time: dash-separated date, space, then time
time_array2 = pd.to_datetime(
    pd.Series(arrays_dict['Time']).astype(str).str.strip(),
    format='%d-%m-%Y %H:%M:%S',  # matches '09-12-2023 21:24:09'
    errors='coerce'
)
# Helper to normalize numbers: strip, remove non-breaking spaces, swap comma->dot
def to_num(series):
    s = pd.Series(series).astype(str).str.strip()
    s = s.str.replace('\u00A0', '', regex=False)  # just in case NBSPs exist
    s = s.str.replace(',', '.', regex=False)      # decimal comma -> dot
    s = s.replace({'': None})                     # empty -> None
    return pd.to_numeric(s, errors='coerce')

delta_time_array2 = to_num(arrays_dict['Delta Time'])
pressure_array2 = to_num(arrays_dict['Pressure']) * 0.0689476 # psi to bar
temperature_array2 = to_num(arrays_dict['Temperature'])

# Optional: drop rows where any key field failed to parse
mask = time_array2.notna() & pressure_array2.notna() & temperature_array2.notna()
time_array2        = time_array2[mask]
delta_time_array2  = delta_time_array2[mask]
pressure_array2    = pressure_array2[mask]
temperature_array2 = temperature_array2[mask]

#mask = (
#    ~np.isnan(pressure_array2) &
#    pd.notna(time_array2)
#)

#time_array2 = time_array2
#delta_time_array2 = delta_time_array2[mask]
#start = pd.to_datetime(time_array.min().strftime('%Y-%m-%d') + ' 15:15:00')
#end = pd.to_datetime(time_array.min().strftime('%Y-%m-%d') + ' 17:30:00')
#pressure_array2 = pressure_array2[mask]
#temperature_array2 = temperature_array2[mask]

print(time_array2)
print(delta_time_array2)
print(pressure_array2)
print(temperature_array2)

# ---------- Figure 3: Pressure & Temperature vs Time ----------
#start, end = time_array2.min(), time_array2.max()

# --- Right before plotting Figure 3, build safe x-limits from time_array2 ---
if time_array2.empty:
    raise ValueError("time_array2 is empty after masking; cannot set x-limits.")

# target 14:00–17:00 on the first day present in time_array2
date0 = time_array2.min().normalize()
start3 = date0 + pd.Timedelta(hours=14)
end3   = date0 + pd.Timedelta(hours=17)

# if that window contains no samples, fall back to full span
has_points = ((time_array2 >= start3) & (time_array2 <= end3)).any()
if not has_points:
    start3, end3 = time_array2.min(), time_array2.max()


fig3, ax4 = plt.subplots(figsize=(16, 6))
ax4.plot(time_array2, pressure_array2, color='royalblue', linestyle='-', linewidth=1, label='Pressure [bar]')
ax4.set_xlabel('Time', fontsize=8)
ax4.set_ylabel('Pressure [bar]', fontsize=8)
ax4.set_title('Pressure and Temperature vs Time', fontsize=8)
ax4.tick_params(axis='both', labelsize=7)
ax4.set_xlim(start3, end3)
ax4.grid()

ax5 = ax4.twinx()
ax5.plot(time_array2, temperature_array2, color='forestgreen', linestyle='-', linewidth=1, label='Temperature [°C]')
ax5.set_ylabel('Temperature [°C]', fontsize=8)
ax5.tick_params(axis='y', labelsize=7)

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

lines4, labels4 = ax4.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()
ax4.legend(lines4 + lines5, labels4 + labels5, fontsize=6, loc='upper left')

fig3.autofmt_xdate()
plt.tight_layout()
plt.show()





