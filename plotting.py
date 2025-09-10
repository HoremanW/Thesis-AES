# plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
def plot_srt(x_sqrt, p_srt, dpdx, i_cl=None, *, figsize=(12, 7)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(x_sqrt, p_srt, lw=1)
    axes[0].set_ylabel("P @ fracture [bar]")
    axes[0].set_title("Closure analysis — Square-root-of-time (downhole clock)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(x_sqrt, dpdx, lw=1)
    axes[1].set_xlabel("sqrt(time since shut-in) [s^0.5]")
    axes[1].set_ylabel("dP/d√t [bar / s^0.5]")
    axes[1].grid(True, alpha=0.4)

    if i_cl is not None:
        for a in axes:
            a.axvline(x_sqrt[i_cl], ls='--')

    fig.tight_layout()
    return fig, axes

def plot_bourdet(t_log, dP_dlogt, x_sqrt=None, i_cl=None, *, figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogx(t_log, dP_dlogt, lw=1)
    ax.set_xlabel("time since shut-in [s] (log) — downhole clock")
    ax.set_ylabel("dP/d(ln t) [bar]")
    ax.set_title("Closure analysis — Bourdet derivative (downhole clock)")
    ax.grid(True, which='both', alpha=0.4)

    if i_cl is not None and x_sqrt is not None:
        ax.axvline(x_sqrt[i_cl]**2, ls='--')

    fig.tight_layout()
    return fig, ax
