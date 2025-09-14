# xlot_runner.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Your local modules (unchanged) ---
import time_difference
import well_corrections
import closure_analysis
import plotting


# ----------------------------
# Utilities
# ----------------------------
def to_num(series: pd.Series) -> pd.Series:
    s = pd.Series(series, copy=False)
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip()
    s = s.str.replace("\u00A0", "", regex=False)  # NBSP
    s = s.str.replace(",", ".", regex=False)      # decimal comma -> dot
    s = s.replace({"": None})
    return pd.to_numeric(s, errors="coerce")


def first_existing(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm = {re.sub(r"\W+", "", c).lower(): c for c in df.columns}
    for a in aliases:
        key = re.sub(r"\W+", "", a).lower()
        if key in norm:
            return norm[key]
    # looser regex
    for a in aliases:
        pat = re.compile(a, re.I)
        for c in df.columns:
            if pat.fullmatch(c) or pat.search(c):
                return c
    return None


def parse_time_any(series: pd.Series, fmts: List[str]) -> pd.Series:
    s = pd.Series(series, copy=False).astype(str).str.strip()
    for f in fmts:
        t = pd.to_datetime(s, format=f, errors="coerce")
        if t.notna().sum() > 0:
            return t
    return pd.to_datetime(s, errors="coerce")


def safe_stem(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    root: Path = Path("Data/Amstelland")
    glob_surface: str = "*Cementing*Unit*Data*.txt"
    glob_downhole: str = "*Downhole*Gauge*Data*.TXT"

    # Plot toggles (speed up batch)
    plots_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "surface_panels": True,
        "surface_triple": True,
        "downhole_PT": True,
        "alignment": True,
        "SRT": True,
        "Bourdet": True,
        "Barree": True,
        "qperf": True,
    })

    # Time formats to try
    time_formats_surface: List[str] = field(default_factory=lambda: [
        "%m:%d:%Y:%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"
    ])
    time_formats_downhole: List[str] = field(default_factory=lambda: [
        "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m:%d:%Y:%H:%M:%S"
    ])

    # Column aliases (extend as needed)
    aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "time":          ["Time", "Timestamp", "DateTime", "DATE TIME", "DATE/TIME"],
        "pressure_S":    ["Treating Pressure", "Pressure (Surface)", "Pressure", "Pressure [bar]"],
        "flow_S":        ["Flow Rate", "Pump Rate", "Rate", "Q", "Flow"],
        "density_S":     ["Density", "Fluid Density", "ρ", "RHO", "Mud Density"],
        "volume_S":      ["Volume", "Pumped Volume", "Total Volume", "Cum Volume", "VOLUME"],
        "tot_vol_nrd_S": ["TOT vol nrd", "TOT_Vol_NRD", "Total Vol NRD"],
        "vol_nrd_S":     ["NRD VOLUME", "Vol NRD", "NRD_Volume"],
        "return_vol_S":  ["RETURN VOL", "Return Volume", "Returned Volume"],
        "tot_balance_S": ["total bla", "Total Balance", "Balance Volume"],
        "rda_S":         ["RDA2 IN1", "RDA", "Aux Pressure", "Backup Pressure"],
        "well_name":     ["WELL", "Well", "Well Name", "WELL NAME", "WELLNAME"],

        "time_D":        ["Time", "Timestamp", "DateTime"],
        "delta_t_D":     ["Delta Time", "Δt", "DeltaTime", "Elapsed"],
        "pressure_D":    ["Pressure", "Gauge Pressure", "P_dh", "Pressure [psi]", "P (psi)"],
        "temperature_D": ["Temperature", "Temp", "T", "Temperature [C]"],
        "well_name_D":   ["WELL", "Well", "Well Name", "WELL NAME", "WELLNAME"],
    })

    # Filters & units
    rda_max: float = 2000.0
    downhole_psi_to_bar: float = 0.0689476

    # Optional windows (None = auto full)
    surface_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    downhole_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    # Lag / hydrostatic / geometry
    max_lag_s: int = 4 * 3600
    detrend_window_s: int = 120
    MD: np.ndarray = field(default_factory=lambda: np.array([1364, 1383, 1383.3], float))
    TVD: np.ndarray = field(default_factory=lambda: np.array([1363.18, np.nan, 1382.31], float))
    TVD_fracture_m: float = 1866.50
    gauge_index: int = 1

    # Shut-in detection
    shutin_threshold_m3h: float = 0.1
    shutin_min_hold_s: int = 30

    # Pre-breakdown PV window (None → auto)
    pre_breakdown_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    # Output
    save_figs: bool = True
    outdir: Path = Path("figures")
    results_csv: Path = Path("figures") / "xlot_results.csv"
    results_parquet: Path = Path("figures") / "xlot_results.parquet"
    dpi: int = 200


# ----------------------------
# Data adapters
# ----------------------------
class SurfaceData:
    def __init__(self, path: Path, cfg: Config):
        self.path = path
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None

        self.time: Optional[pd.Series] = None
        self.p: Optional[pd.Series] = None
        self.q: Optional[pd.Series] = None
        self.rho: Optional[pd.Series] = None
        self.vol: Optional[pd.Series] = None
        self.return_vol: Optional[pd.Series] = None
        self.total_balance: Optional[pd.Series] = None
        self.rda: Optional[pd.Series] = None

        # metadata
        self.well_name: Optional[str] = None
        self.date_hint: Optional[pd.Timestamp] = None

    def load(self) -> "SurfaceData":
        df = pd.read_csv(self.path, sep="\t", engine="python", on_bad_lines="skip")
        # drop a header/units row if row 1 parses as date but row 0 doesn't (heuristic)
        if df.shape[0] > 1 and pd.to_datetime(df.iloc[0, 0], errors="coerce") is pd.NaT and pd.to_datetime(df.iloc[1, 0], errors="coerce") is not pd.NaT:
            df = df.iloc[1:].reset_index(drop=True)
        self.df = df
        self._parse_and_filter()
        self._extract_metadata()
        return self

    def _col(self, key: str) -> pd.Series:
        assert self.df is not None
        name = first_existing(self.df, self.cfg.aliases[key])
        return to_num(self.df[name]) if name else pd.Series([np.nan] * len(self.df))

    def _parse_and_filter(self) -> None:
        df = self.df; assert df is not None
        tcol = first_existing(df, self.cfg.aliases["time"])
        t = parse_time_any(df[tcol], self.cfg.time_formats_surface) if tcol else pd.Series(pd.NaT, index=df.index)

        p = self._col("pressure_S")
        q = self._col("flow_S")
        rho = self._col("density_S")
        vol = self._col("volume_S")
        tot_nrd = self._col("tot_vol_nrd_S")
        vol_nrd = self._col("vol_nrd_S")
        ret = self._col("return_vol_S")
        bal = self._col("tot_balance_S")
        rda = self._col("rda_S")

        mask = t.notna() & p.notna()
        if rda.notna().any():
            mask &= (rda.fillna(np.inf) <= self.cfg.rda_max)
        if self.cfg.surface_window:
            s0, s1 = self.cfg.surface_window
            mask &= (t >= s0) & (t <= s1)

        self.time = t[mask]
        self.p = p[mask]
        self.q = q[mask]
        self.rho = rho[mask]
        self.vol = vol[mask]
        self.return_vol = ret[mask]
        self.total_balance = bal[mask]
        self.rda = rda[mask]

    def _extract_metadata(self) -> None:
        assert self.df is not None
        # well name
        wcol = first_existing(self.df, self.cfg.aliases["well_name"])
        if wcol:
            # take most frequent non-null value as name
            vals = self.df[wcol].dropna().astype(str).str.strip()
            self.well_name = vals.mode().iloc[0] if not vals.empty else None
        if not self.well_name:
            # derive from filename (e.g., AMS-01, or token before first underscore)
            m = re.search(r"[A-Z]{2,}-\d{1,3}", self.path.name)
            self.well_name = m.group(0) if m else re.split(r"[_\-]", self.path.stem, maxsplit=1)[0]

        # date hint
        if self.time is not None and self.time.notna().any():
            self.date_hint = pd.to_datetime(self.time.min()).normalize()
        else:
            self.date_hint = None


class DownholeData:
    def __init__(self, path: Path, cfg: Config):
        self.path = path
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None

        self.time: Optional[pd.Series] = None
        self.delta_t: Optional[pd.Series] = None
        self.p_bar: Optional[pd.Series] = None
        self.temp: Optional[pd.Series] = None

        # metadata
        self.well_name: Optional[str] = None
        self.date_hint: Optional[pd.Timestamp] = None

    def load(self) -> "DownholeData":
        df = pd.read_csv(self.path, sep="\t", engine="python", on_bad_lines="skip")
        # Try to trim header rows typical for downhole export
        if df.shape[0] > 4 and pd.to_datetime(str(df.iloc[0, 0]).strip(), errors="coerce") is pd.NaT and pd.to_datetime(str(df.iloc[4, 0]).strip(), errors="coerce") is not pd.NaT:
            df = df.iloc[4:, :].reset_index(drop=True)
        # drop junk tail columns (if present)
        if df.shape[1] >= 6 and {"None", "None2"} & set(map(str, df.columns)):
            df = df.iloc[:, :4]
        self.df = df
        self._parse_and_filter()
        self._extract_metadata()
        return self

    def _col(self, key: str) -> pd.Series:
        assert self.df is not None
        name = first_existing(self.df, self.cfg.aliases[key])
        return to_num(self.df[name]) if name else pd.Series([np.nan] * len(self.df))

    def _parse_and_filter(self) -> None:
        df = self.df; assert df is not None
        tcol = first_existing(df, self.cfg.aliases["time_D"])
        t = parse_time_any(df[tcol].astype(str).str.strip(), self.cfg.time_formats_downhole) if tcol else pd.Series(pd.NaT, index=df.index)

        dts = self._col("delta_t_D")
        p_raw = self._col("pressure_D")
        temp = self._col("temperature_D")

        # auto psi→bar heuristic
        p_bar = p_raw.copy()
        if np.nanmedian(p_raw) > 100:
            p_bar = p_raw * self.cfg.downhole_psi_to_bar

        mask = t.notna() & p_bar.notna() & temp.notna()
        if self.cfg.downhole_window:
            d0, d1 = self.cfg.downhole_window
            mask &= (t >= d0) & (t <= d1)

        self.time = t[mask]
        self.delta_t = dts[mask]
        self.p_bar = p_bar[mask]
        self.temp = temp[mask]

    def _extract_metadata(self) -> None:
        assert self.df is not None
        wcol = first_existing(self.df, self.cfg.aliases["well_name_D"])
        if wcol:
            vals = self.df[wcol].dropna().astype(str).str.strip()
            self.well_name = vals.mode().iloc[0] if not vals.empty else None
        if not self.well_name:
            m = re.search(r"[A-Z]{2,}-\d{1,3}", self.path.name)
            self.well_name = m.group(0) if m else re.split(r"[_\-]", self.path.stem, maxsplit=1)[0]

        if self.time is not None and self.time.notna().any():
            self.date_hint = pd.to_datetime(self.time.min()).normalize()
        else:
            self.date_hint = None


# ----------------------------
# End-to-end job
# ----------------------------
class XLOTJob:
    def __init__(self, surface: SurfaceData, downhole: DownholeData, cfg: Config):
        self.surf = surface
        self.dh = downhole
        self.cfg = cfg
        self.lag_s: float = 0.0
        self.delta_tvd_m: float = 0.0
        self.ts_dh: Optional[np.ndarray] = None
        self.p_dh: Optional[np.ndarray] = None

        # results to report
        self.result_row: Dict[str, object] = {}

    def run(self) -> None:
        # Depth & lag prerequisites
        _, TVD_gauge_m, delta_tvd_m = well_corrections.estimate_lag(
            self.cfg.MD, self.cfg.TVD, self.cfg.gauge_index, self.cfg.TVD_fracture_m
        )
        self.delta_tvd_m = float(delta_tvd_m)

        lag_s, _ = time_difference.estimate_delay_seconds_robust(
            self.surf.time, self.surf.p,
            self.dh.time, self.dh.p_bar,
            max_lag_s=self.cfg.max_lag_s, detrend_window_s=self.cfg.detrend_window_s
        )
        self.lag_s = float(lag_s)
        print(f"[{self.surf.path.name}] Estimated delay surface→downhole: {self.lag_s/3600:.2f} h")

        # Hydrostatic corrections to fracture depth
        p_surface_corr, _ = well_corrections.hydrostatic_correct_to_fracture(
            p_gauge=self.surf.p, time_gauge=self.surf.time,
            rho_surface=self.surf.rho, time_surface=self.surf.time,
            delta_tvd_m=self.cfg.TVD_fracture_m, out_units="bar", lag_s=None
        )
        p_downhole_corr, _ = well_corrections.hydrostatic_correct_to_fracture(
            p_gauge=self.dh.p_bar, time_gauge=self.dh.time,
            rho_surface=self.surf.rho, time_surface=self.surf.time,
            delta_tvd_m=self.delta_tvd_m, out_units="bar", lag_s=self.lag_s
        )

        # Shut-in detection (surface clock) → convert to DH clock
        t_shut_in_surf = closure_analysis.find_shut_in_time_from_flow(
            self.surf.time, self.surf.q,
            threshold=self.cfg.shutin_threshold_m3h, min_hold_s=self.cfg.shutin_min_hold_s
        )
        if t_shut_in_surf is None:
            t_shut_in_surf = self.surf.time.iloc[-1]
        t_shut_in_dh = t_shut_in_surf - pd.to_timedelta(self.lag_s, unit="s")

        # Downhole falloff arrays (Δt since shut-in, P)
        ts_dh, p_dh = closure_analysis.build_shut_in_series(self.dh.time, p_downhole_corr, t_shut_in_dh)
        self.ts_dh, self.p_dh = ts_dh, p_dh

        # ISIP (first post-shut-in sample)
        ISIP_bar = float(p_dh[0]) if len(p_dh) else np.nan

        # SRT & closure pick
        x_sqrt, p_srt, dpdx = closure_analysis.derivative_vs_sqrt_time(ts_dh, p_dh)
        i_cl = closure_analysis.suggest_closure_from_srt(x_sqrt, p_srt, dpdx, min_t_s=90, guard_s=3)
        if i_cl is not None:
            closure_time_s = float(x_sqrt[i_cl] ** 2)
            closure_pressure_bar = float(p_srt[i_cl])
        else:
            closure_time_s = np.nan
            closure_pressure_bar = np.nan

        # Estimate pumping duration tp (first sustained flow → shut-in)
        t0 = self._first_sustained_pump_start(self.surf.time, self.surf.q, threshold=0.1, hold_s=30)
        if t0 is None:
            t0 = t_shut_in_surf - pd.Timedelta(minutes=30)
        tp_seconds = (t_shut_in_surf - t0).total_seconds()

        # PV compliance & q@perfs
        if self.cfg.pre_breakdown_window:
            pre0, pre1 = self.cfg.pre_breakdown_window
            mask_pre = (self.surf.time >= pre0) & (self.surf.time <= pre1)
        else:
            pre1 = t_shut_in_surf - pd.Timedelta(minutes=10)
            pre0 = pre1 - pd.Timedelta(minutes=20)
            mask_pre = (self.surf.time >= pre0) & (self.surf.time <= pre1)

        C_well, stats = well_corrections.estimate_wellbore_compliance_from_pv(
            self.surf.time, self.surf.p, self.surf.vol, mask=mask_pre
        )

        # ---- RESULTS ROW ----
        self.result_row = {
            "surface_file": str(self.surf.path),
            "downhole_file": str(self.dh.path),
            "well_name_surface": self.surf.well_name,
            "well_name_downhole": self.dh.well_name,
            "pair_key": self._pair_key(),
            "date_surface_min": pd.to_datetime(self.surf.time.min()) if self.surf.time is not None else pd.NaT,
            "date_surface_max": pd.to_datetime(self.surf.time.max()) if self.surf.time is not None else pd.NaT,
            "date_downhole_min": pd.to_datetime(self.dh.time.min()) if self.dh.time is not None else pd.NaT,
            "date_downhole_max": pd.to_datetime(self.dh.time.max()) if self.dh.time is not None else pd.NaT,
            "lag_s": self.lag_s,
            "tp_seconds": tp_seconds,
            "ISIP_bar": ISIP_bar,
            "closure_time_s": closure_time_s,
            "closure_pressure_bar": closure_pressure_bar,
            "C_well_m3_per_bar": float(C_well) if np.isfinite(C_well) else np.nan,
            "C_well_fit_n": int(stats.get("n", 0)),
            "C_well_fit_r2": float(stats.get("r2", np.nan)),
            "shut_in_surface": pd.to_datetime(t_shut_in_surf),
            "shut_in_downhole": pd.to_datetime(t_shut_in_dh),
        }

        # ---- PLOTS (optional per config) ----
        if self.cfg.plots_enabled.get("surface_panels", True):
            fig1, _ = plotting.plot_surface_panels(
                self.surf.time, self.surf.p, self.surf.q, self.surf.rho, self.surf.vol,
                self.surf.vol, self.surf.return_vol, self.surf.total_balance, self.surf.rda,
                figsize=(22, 8), markersize=0.2, linewidth=1.0, tick_labelsize=7, title_labelsize=6
            )
            self._save_fig(fig1, "01_surface_panels")

        if self.cfg.plots_enabled.get("surface_triple", True):
            fig2, _ = plotting.plot_triple_axis(
                self.surf.time, self.surf.p, self.surf.q, self.surf.return_vol,
                start=self.surf.time.min(), end=self.surf.time.max(), figsize=(16, 6)
            )
            self._save_fig(fig2, "02_surface_triple")

        if self.cfg.plots_enabled.get("downhole_PT", True):
            fig3, _ = plotting.plot_downhole_pt(self.dh.time, self.dh.p_bar, self.dh.temp, figsize=(16, 6))
            self._save_fig(fig3, "03_downhole_PT")

        if self.cfg.plots_enabled.get("alignment", True):
            ts_surface_aligned = pd.to_datetime(self.surf.time) - pd.to_timedelta(self.lag_s, unit="s")
            figA, _ = plotting.plot_alignment(ts_surface_aligned, p_surface_corr, self.dh.time, p_downhole_corr, figsize=(16, 6))
            self._save_fig(figA, "04_alignment")

        if self.cfg.plots_enabled.get("SRT", True):
            figSRT, axesSRT = plotting.plot_srt(x_sqrt, p_srt, dpdx)
            if i_cl is not None:
                for ax in axesSRT:
                    ax.axvline(x_sqrt[i_cl], ls="--")
            self._save_fig(figSRT, "05_SRT")

        if self.cfg.plots_enabled.get("Bourdet", True):
            t_log, dP_dlogt = closure_analysis.bourdet_derivative(ts_dh, p_dh, smooth_win=None)
            figBRD, _ = plotting.plot_bourdet(t_log, dP_dlogt, x_sqrt=x_sqrt, i_cl=i_cl)
            self._save_fig(figBRD, "06_Bourdet")

        if self.cfg.plots_enabled.get("Barree", True):
            figs_barree = plotting.plot_barree_figs_normal_leakoff(
                t_post_s=ts_dh, p_post=p_dh, tp_seconds=tp_seconds,
                auto_closure=True, auto_fit_frac=0.25, auto_tol=0.10
            )
            for name, fig in figs_barree.items():
                self._save_fig(fig, f"07_{name}")

        if self.cfg.plots_enabled.get("qperf", True):
            q_perf_S, parts = well_corrections.flow_at_perfs_from_surface(
                q_pump=self.surf.q, time_q=self.surf.time, pressure_bar=self.surf.p, time_p=self.surf.time,
                C_well_m3_per_bar=float(C_well), out_units="m3/h", smooth_dpdt_s=5
            )
            fig_q, axq = plt.subplots(figsize=(14, 5))
            axq.plot(self.surf.time, self.surf.q, label="q_pump [m³/h]", lw=1)
            axq.plot(self.surf.time, q_perf_S,   label="q_perf @ perfs [m³/h]", lw=1)
            axq.set_ylabel("Flowrate [m³/h]"); axq.grid(True, alpha=0.4)
            axq.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axq2 = axq.twinx()
            axq2.plot(self.surf.time, (parts["dpdt_bar_per_s_on_q"] * float(C_well) * 3600.0), ls="--", lw=1, label="C_well·dp/dt [m³/h]")
            axq2.set_ylabel("C_well·dp/dt [m³/h]")
            l1, lb1 = axq.get_legend_handles_labels(); l2, lb2 = axq2.get_legend_handles_labels()
            axq.legend(l1 + l2, lb1 + lb2, loc="upper center", fontsize=8)
            fig_q.autofmt_xdate(); fig_q.tight_layout()
            self._save_fig(fig_q, "08_qperf")

    def _pair_key(self) -> str:
        # normalized well key for results table
        s = (self.surf.well_name or "").upper()
        d = (self.dh.well_name or "").upper()
        if s and d and s == d:
            return s
        return s or d or safe_stem(self.surf.path)

    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        if not self.cfg.save_figs:
            plt.close(fig)
            return
        outdir = self.cfg.outdir / safe_stem(self.surf.path)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png", dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _first_sustained_pump_start(time_series, flow_series, threshold=0.1, hold_s=30):
        t = pd.to_datetime(pd.Series(time_series))
        q = pd.Series(flow_series, dtype=float)
        for i in range(len(q)):
            if pd.notna(q.iloc[i]) and q.iloc[i] > threshold:
                t0 = t.iloc[i]; j = i; ok = True
                while j < len(q) and (t.iloc[j] - t0).total_seconds() <= hold_s:
                    if pd.notna(q.iloc[j]) and q.iloc[j] <= threshold:
                        ok = False; break
                    j += 1
                if ok: return t0
        return None


# ----------------------------
# Batch runner with smarter pairing
# ----------------------------
class Batch:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.results: List[Dict[str, object]] = []

    def discover(self) -> List[Tuple[Path, Path]]:
        surf_files = sorted(self.cfg.root.glob(self.cfg.glob_surface))
        dh_files   = sorted(self.cfg.root.glob(self.cfg.glob_downhole))
        if not surf_files or not dh_files:
            return []

        # Parse quick metadata (well key + date hint) for all files
        surf_meta = []
        for s in surf_files:
            S = SurfaceData(s, self.cfg).load()
            surf_meta.append((s, (S.well_name or ""), S.date_hint))

        dh_meta = []
        for d in dh_files:
            D = DownholeData(d, self.cfg).load()
            dh_meta.append((d, (D.well_name or ""), D.date_hint))

        # Build pairs: match by same well key first; within that, minimal date difference
        pairs: List[Tuple[Path, Path]] = []
        used_dh: set[Path] = set()

        def norm_key(k: str) -> str:
            return re.sub(r"\W+", "", (k or "").upper())

        dh_by_key: Dict[str, List[Tuple[Path, Optional[pd.Timestamp]]]] = {}
        for d_path, wkey, ddate in dh_meta:
            dh_by_key.setdefault(norm_key(wkey), []).append((d_path, ddate))

        for s_path, wkey_s, sdate in surf_meta:
            candidates = dh_by_key.get(norm_key(wkey_s), [])
            pick: Optional[Path] = None
            if candidates:
                # choose by nearest date hint
                def diff_days(dd):
                    if sdate is None or dd is None:
                        return 999999
                    return abs((sdate - dd).days)
                pick = sorted(candidates, key=lambda x: diff_days(x[1]))[0][0]
                if pick in used_dh:
                    pick = None

            # fallback: nearest date across all DH
            if pick is None:
                left = [(d, ddate) for (d, w, ddate) in dh_meta if d not in used_dh]
                if left:
                    def diff_days_any(dd):
                        if sdate is None or dd is None:
                            return 999999
                        return abs((sdate - dd).days)
                    pick = sorted(left, key=lambda x: diff_days_any(x[1]))[0][0]

            if pick:
                used_dh.add(pick)
                pairs.append((s_path, pick))

        # last fallback: zip any remaining if sizes match
        if not pairs:
            for s, d in zip(surf_files, dh_files):
                pairs.append((s, d))
        return pairs

    def run(self) -> None:
        pairs = self.discover()
        if not pairs:
            print(f"No pairs found under {self.cfg.root}. Adjust glob patterns in Config.")
            return

        self.cfg.outdir.mkdir(parents=True, exist_ok=True)

        for s_path, d_path in pairs:
            print(f"\n=== Processing ===\nSurface : {s_path.name}\nDownhole: {d_path.name}")
            S = SurfaceData(s_path, self.cfg).load()
            D = DownholeData(d_path, self.cfg).load()

            job = XLOTJob(S, D, self.cfg)
            job.run()
            self.results.append(job.result_row)

        # Write result table
        self._write_results()

    def _write_results(self) -> None:
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        df.sort_values(["pair_key", "date_surface_min"], inplace=True)

        # CSV
        self.cfg.results_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cfg.results_csv, index=False)
        print(f"Saved CSV results → {self.cfg.results_csv}")

        # Parquet (optional)
        try:
            df.to_parquet(self.cfg.results_parquet, index=False)
            print(f"Saved Parquet results → {self.cfg.results_parquet}")
        except Exception as e:
            print(f"Parquet save skipped (install pyarrow or fastparquet). Reason: {e}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    cfg = Config(
        root=Path("Data/Amstelland"),
        save_figs=True,
        outdir=Path("figures"),
        results_csv=Path("figures") / "xlot_results.csv",
        results_parquet=Path("figures") / "xlot_results.parquet",
        # Example: skip heavy plots in big batches
        # plots_enabled={"surface_panels": False, "surface_triple": False, "downhole_PT": True,
        #               "alignment": False, "SRT": True, "Bourdet": True, "Barree": True, "qperf": False}
    )
    Batch(cfg).run()
