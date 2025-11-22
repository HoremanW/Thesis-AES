from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# -----------------------------
# Result container for Haimson
# -----------------------------
@dataclass
class HaimsonResult:
    P_sorted: np.ndarray        # full sorted P (including skipped highs)
    Y_sorted: np.ndarray        # full sorted Y = -dP/dt
    Y_fit: np.ndarray           # fitted bilinear curve at P_sorted
    m1: float                   # slope low-pressure segment
    m2: float                   # slope high-pressure segment
    P_break: float              # break pressure (Ps)
    P_anchor: float             # anchor pressure (lowest P used in fit)
    Y_anchor: float             # anchor Y
    Y_break: float              # Y at P_break
    R2_all: float               # overall R²
    R2_left: float              # R² on left segment
    R2_right: float             # R² on right segment


# --------------------------------------
# Class: Haimson-style bilinear picker
# --------------------------------------
class HaimsonBilinearPicker:
    """
    Haimson-style bilinear FCP picker.

    Uses Y = -dP/dt vs P with a continuous bilinear model anchored at the
    lowest-P datapoint. Wraps your `haimson_bilinear_fit_anchored` logic
    into a reusable class.
    """

    def __init__(self, min_seg_size: int = 5, skip_high_n: int = 3):
        """
        Parameters
        ----------
        min_seg_size : int
            Minimum number of points per segment (left/right of break).
        skip_high_n : int
            Number of highest-pressure points to exclude from the fit
            (typically to avoid noisy tail).
        """
        self.min_seg_size = min_seg_size
        self.skip_high_n = skip_high_n

    # --- static helper: anchored bilinear model ---
    @staticmethod
    def bilinear_model_anchored(P, m1, m2, P_break, P_anchor, Y_anchor):
        """
        Continuous bilinear model with the LEFT segment anchored at (P_anchor, Y_anchor):

            For P <= P_break:  Y = Y_break + m1 * (P - P_break)
            For P >  P_break:  Y = Y_break + m2 * (P - P_break)

        with Y_break determined by requiring the left segment to pass exactly through
        (P_anchor, Y_anchor):

            Y_anchor = Y_break + m1 * (P_anchor - P_break)
            -> Y_break = Y_anchor - m1 * (P_anchor - P_break)
        """
        P = np.asarray(P, dtype=float)

        Y_break = Y_anchor - m1 * (P_anchor - P_break)

        Y = np.empty_like(P)
        mask_left = P <= P_break
        mask_right = ~mask_left

        Y[mask_left] = Y_break + m1 * (P[mask_left] - P_break)
        Y[mask_right] = Y_break + m2 * (P[mask_right] - P_break)

        return Y

    # --- core NL fit, equivalent to haimson_bilinear_fit_anchored ---
    def fit_PY(self, P: np.ndarray, Y: np.ndarray) -> Optional[HaimsonResult]:
        """
        Fit Y vs P with anchored bilinear model.

        Returns
        -------
        HaimsonResult or None if fit fails / insufficient data.
        """
        P = np.asarray(P, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # drop NaNs / infs
        mask = np.isfinite(P) & np.isfinite(Y)
        P = P[mask]
        Y = Y[mask]

        if P.size == 0:
            return None

        # sort by pressure
        sort_idx = np.argsort(P)
        P_sorted_all = P[sort_idx]
        Y_sorted_all = Y[sort_idx]

        # optionally drop highest-P points for the fit
        if self.skip_high_n > 0:
            if P_sorted_all.size <= self.skip_high_n:
                return None
            P_sorted = P_sorted_all[:-self.skip_high_n]
            Y_sorted = Y_sorted_all[:-self.skip_high_n]
        else:
            P_sorted = P_sorted_all
            Y_sorted = Y_sorted_all

        n = P_sorted.size
        if n < 2 * self.min_seg_size + 2:
            return None

        # anchor point: lowest-P datapoint after skipping highs
        P_anchor = P_sorted[0]
        Y_anchor = Y_sorted[0]

        # allowed breakpoint range
        P_min_allowed = P_sorted[self.min_seg_size]
        P_max_allowed = P_sorted[-self.min_seg_size - 1]
        if P_min_allowed >= P_max_allowed:
            return None

        # crude slope guesses
        m1_init, _ = np.polyfit(
            P_sorted[: self.min_seg_size + 2], Y_sorted[: self.min_seg_size + 2], 1
        )
        m2_init, _ = np.polyfit(
            P_sorted[-(self.min_seg_size + 2) :], Y_sorted[-(self.min_seg_size + 2) :], 1
        )

        P_break_init = 0.5 * (P_min_allowed + P_max_allowed)
        p0 = [m1_init, m2_init, P_break_init]
        lower_bounds = [-np.inf, -np.inf, P_min_allowed]
        upper_bounds = [np.inf, np.inf, P_max_allowed]

        def model_for_fit(Px, m1, m2, P_break):
            return self.bilinear_model_anchored(Px, m1, m2, P_break, P_anchor, Y_anchor)

        try:
            popt, pcov = curve_fit(
                model_for_fit,
                P_sorted,
                Y_sorted,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
        except Exception as e:
            print("Haimson anchored bilinear fit failed:", e)
            return None

        m1, m2, P_break = popt

        # Y_break from anchor condition
        Y_break = Y_anchor - m1 * (P_anchor - P_break)

        # fit over subset
        Y_fit_subset = model_for_fit(P_sorted, *popt)
        ss_res = np.sum((Y_sorted - Y_fit_subset) ** 2)
        ss_tot = np.sum((Y_sorted - Y_sorted.mean()) ** 2)
        R2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # segment-wise R²
        mask_left = P_sorted <= P_break
        mask_right = ~mask_left

        def segment_R2(P_seg, Y_seg, m, P_break, Y_break):
            if P_seg.size < 2:
                return np.nan
            Y_pred = Y_break + m * (P_seg - P_break)
            ss_res_seg = np.sum((Y_seg - Y_pred) ** 2)
            ss_tot_seg = np.sum((Y_seg - Y_seg.mean()) ** 2)
            return 1.0 - ss_res_seg / ss_tot_seg if ss_tot_seg > 0 else np.nan

        R2_left = segment_R2(
            P_sorted[mask_left], Y_sorted[mask_left], m1, P_break, Y_break
        )
        R2_right = segment_R2(
            P_sorted[mask_right], Y_sorted[mask_right], m2, P_break, Y_break
        )

        # Evaluate full fit on all sorted points (including skipped highs)
        Y_fit_full = self.bilinear_model_anchored(
            P_sorted_all, m1, m2, P_break, P_anchor, Y_anchor
        )

        return HaimsonResult(
            P_sorted=P_sorted_all,
            Y_sorted=Y_sorted_all,
            Y_fit=Y_fit_full,
            m1=m1,
            m2=m2,
            P_break=P_break,
            P_anchor=P_anchor,
            Y_anchor=Y_anchor,
            Y_break=Y_break,
            R2_all=R2_all,
            R2_left=R2_left,
            R2_right=R2_right,
        )

    # --- helper: take a DF cycle (your G_cycles[cyc_id]) and do full Haimson analysis ---
    def pick_from_cycle_df(self, df_cycle) -> Optional[HaimsonResult]:
        """
        Given one G_cycles[cyc_id] DataFrame with columns:
            - 'delta_t_s'
            - 'P_dh_at_frac_bar'
        compute Y = -dP/dt and fit Haimson bilinear model.

        Returns HaimsonResult or None.
        """
        dt_s = df_cycle["delta_t_s"].to_numpy()
        P = df_cycle["P_dh_at_frac_bar"].to_numpy()

        # finite only
        mask = np.isfinite(dt_s) & np.isfinite(P)
        dt_s = dt_s[mask]
        P = P[mask]

        if dt_s.size < 5:
            return None

        # ensure time increasing
        sort_idx_t = np.argsort(dt_s)
        dt_s = dt_s[sort_idx_t]
        P = P[sort_idx_t]

        # derivative dP/dt (bar/sec)
        dPdt = np.gradient(P, dt_s)
        Y = -dPdt  # Haimson uses -dP/dt so decay rate is positive

        return self.fit_PY(P, Y)
    


@dataclass
class StiffnessPick:
    idx_min: int
    idx_target: Optional[int]
    dP_dG_min: float
    dP_dG_target: float
    P_at_min: float
    P_at_target: Optional[float]
    G_at_min: float
    G_at_target: Optional[float]
    t_at_min: Optional[float]
    t_at_target: Optional[float]
    dP_dG: np.ndarray


def system_stiffness_pick(
    G: np.ndarray,
    P: np.ndarray,
    t: Optional[np.ndarray] = None,
    smooth: bool = False,
    smooth_window: int = 5,
) -> StiffnessPick:
    """
    System stiffness pick based on the MINIMUM of -dP/dG:
      - compute dP/dG vs G,
      - define D = -dP/dG,
      - find min(D),
      - define target = 1.10 * min(D),
      - find first point AFTER the minimum where D >= target,
      - report pressure at that point (P_at_target).
    """
    G = np.asarray(G, dtype=float)
    P = np.asarray(P, dtype=float)
    if G.shape != P.shape:
        raise ValueError("G and P must have the same shape.")

    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != G.shape:
            raise ValueError("t must have same shape as G and P.")

    # original derivative
    dP_dG = np.gradient(P, G)

    # apply negative transformation
    neg_dP_dG = -dP_dG

    # Optional smoothing
    if smooth and smooth_window > 1:
        k = smooth_window
        if k % 2 == 0:
            k += 1
        pad = k // 2
        kernel = np.ones(k) / k
        neg_dP_dG = np.convolve(
            np.pad(neg_dP_dG, pad_width=pad, mode="edge"),
            kernel,
            mode="valid"
        )

    # 1) Find minimum of the NEGATIVE derivative
    idx_min = int(np.nanargmin(neg_dP_dG))
    dP_dG_min = float(neg_dP_dG[idx_min])

    # 2) Target = 110% of that minimum
    dP_dG_target = 1.10 * dP_dG_min

    # 3) Find first index AFTER the minimum where neg_dP_dG >= target
    idx_target = None
    for i in range(idx_min + 1, len(neg_dP_dG)):
        if neg_dP_dG[i] >= dP_dG_target:
            idx_target = int(i)
            break

    # 4) Gather results
    P_at_min = float(P[idx_min])
    G_at_min = float(G[idx_min])
    t_at_min = float(t[idx_min]) if t is not None else None

    if idx_target is not None:
        P_at_target = float(P[idx_target])
        G_at_target = float(G[idx_target])
        t_at_target = float(t[idx_target]) if t is not None else None
    else:
        P_at_target = None
        G_at_target = None
        t_at_target = None

    return StiffnessPick(
        idx_min=idx_min,
        idx_target=idx_target,
        dP_dG_min=dP_dG_min,
        dP_dG_target=dP_dG_target,
        P_at_min=P_at_min,
        P_at_target=P_at_target,
        G_at_min=G_at_min,
        G_at_target=G_at_target,
        t_at_min=t_at_min,
        t_at_target=t_at_target,
        dP_dG=neg_dP_dG,   # store the NEGATIVE derivative used for picking
    )

def plot_stiffness_cycle(G, P, t, pick: StiffnessPick, title: str = ""):
    G = np.asarray(G, dtype=float)
    P = np.asarray(P, dtype=float)
    neg_dP_dG = pick.dP_dG  # now NEGATIVE derivative

    fig, ax1 = plt.subplots()

    # Pressure vs G
    ax1.plot(G, P, label="Pressure", alpha=0.8)
    ax1.set_xlabel("G-function")
    ax1.set_ylabel("Pressure [bar]")

    # -dP/dG vs G
    ax2 = ax1.twinx()
    ax2.plot(G, neg_dP_dG, "--", label="-dP/dG (used for pick)", alpha=0.8)
    ax2.set_ylabel("-dP/dG")

    # Mark min point
    ax2.plot(pick.G_at_min, pick.dP_dG_min, "ro", label="min -dP/dG")

    # Mark target point
    if pick.idx_target is not None:
        ax2.plot(pick.G_at_target, pick.dP_dG_target, "go", label="110% min")
        ax1.axvline(pick.G_at_target, color="g", linestyle="--", alpha=0.5)
        ax1.axhline(pick.P_at_target, color="g", linestyle=":", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    if title:
        ax1.set_title(title)

    fig.tight_layout()
    return fig


# ------------------------------
# Dataclass for Castillo results
# ------------------------------
@dataclass
class CastilloCycleResult:
    G: np.ndarray
    P: np.ndarray
    t_s: np.ndarray
    dP_dG: np.ndarray
    idx_open_start: int
    idx_open_end: int
    P_star: float
    P_isip_theoretical: float
    idx_closure: Optional[int]
    P_closure: Optional[float]
    r2_open: float


# -----------------------------------------------------
# Rolling derivative: local linear regression (Castillo)
# -----------------------------------------------------
def castillo_rolling_dP_dG(G: np.ndarray, P: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Rolling dP/dG using local linear regression (Castillo 1987 style).
    """
    G = np.asarray(G, float)
    P = np.asarray(P, float)

    if window % 2 == 0:
        window += 1
    half = window // 2

    d = np.full_like(G, np.nan)

    for i in range(half, len(G) - half):
        g_win = G[i-half:i+half+1]
        p_win = P[i-half:i+half+1]
        A = np.vstack([g_win, np.ones_like(g_win)]).T
        m, _ = np.linalg.lstsq(A, p_win, rcond=None)[0]
        d[i] = m
    return d


# ------------------------------
# Main Castillo analysis (1 cycle)
# ------------------------------
def castillo_analyze_cycle_df(
    df_cycle: pd.DataFrame,
    col_t: str = "delta_t_s",
    col_G: str = "G",
    col_P: str = "P_dh_at_frac_bar",
    open_frac_window=(0.1, 0.5),   # <-- YOU control open-fracture window here
    deriv_window: int = 7,
    deriv_tol_rel: float = 0.10,
    t_min_for_linear: float = 0.0,  # skip very early time if you want
) -> CastilloCycleResult:
    """
    Castillo (1987) G-function analysis on a single cycle.

    open_frac_window = (g_start_frac, g_end_frac) defines the open-fracture window
    as a fraction of the G-range AFTER t >= t_min_for_linear.
    Example: (0.1, 0.5) → use G between 10% and 50% of the G-range.

    Parameters
    ----------
    df_cycle : DataFrame
        Must contain at least: col_t, col_G, col_P.
    col_t, col_G, col_P : str
        Column names for time since shut-in, G-function, and pressure.
    open_frac_window : (float, float)
        Fraction of G-range to define open-fracture region.
    deriv_window : int
        Window length (points) for rolling dP/dG regression.
    deriv_tol_rel : float
        Relative deviation from early-time dP/dG mean to declare closure.
    t_min_for_linear : float
        Minimum Δt (seconds since shut-in) to include in the open-fracture window
        (skips very early ISIP region if > 0).

    Returns
    -------
    CastilloCycleResult
    """
    # 1) Extract arrays
    t_s = df_cycle[col_t].to_numpy(float)   # Δt since shut-in [s]
    G = df_cycle[col_G].to_numpy(float)
    P = df_cycle[col_P].to_numpy(float)

    # NaN-safe
    mask = np.isfinite(t_s) & np.isfinite(G) & np.isfinite(P)
    t_s = t_s[mask]
    G = G[mask]
    P = P[mask]

    if len(G) < max(deriv_window + 5, 10):
        raise ValueError("Not enough points in cycle for Castillo analysis.")

    # 2) Rolling derivative dP/dG
    dP_dG = castillo_rolling_dP_dG(G, P, window=deriv_window)

    # 3) Define open-fracture window in terms of G-range and t_min_for_linear
    G_min, G_max = np.nanmin(G), np.nanmax(G)
    g_start_frac, g_end_frac = open_frac_window
    g_start = G_min + g_start_frac * (G_max - G_min)
    g_end   = G_min + g_end_frac   * (G_max - G_min)

    open_mask = (G >= g_start) & (G <= g_end) & (t_s >= t_min_for_linear)
    open_idx = np.where(open_mask)[0]

    if len(open_idx) < 5:
        raise ValueError(
            f"Open-fracture window has too few points. "
            f"Try different open_frac_window or t_min_for_linear."
        )

    idx_open_start = int(open_idx[0])
    idx_open_end   = int(open_idx[-1])

    # 4) Linear regression P vs G in open-fracture window → P* and theoretical ISIP
    Gw = G[open_mask]
    Pw = P[open_mask]
    A = np.vstack([Gw, np.ones_like(Gw)]).T
    P_star, P_isip = np.linalg.lstsq(A, Pw, rcond=None)[0]

    # Compute R² for this linear fit (for info)
    P_pred = P_star * Gw + P_isip
    ss_res = np.sum((Pw - P_pred)**2)
    ss_tot = np.sum((Pw - np.mean(Pw))**2)
    r2_open = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 5) Closure: first deviation of dP/dG from early-time mean in the open window
    # Use ~first half of open window as "early plateau"
    early_end = idx_open_start + max(5, (idx_open_end - idx_open_start) // 2)
    d_early = dP_dG[idx_open_start:early_end]
    d_early = d_early[np.isfinite(d_early)]
    d_mean = np.mean(d_early) if len(d_early) else np.nan

    idx_closure = None
    P_closure = None

    if np.isfinite(d_mean):
        lower = d_mean * (1.0 - deriv_tol_rel)
        upper = d_mean * (1.0 + deriv_tol_rel)

        for i in range(idx_open_end + 1, len(dP_dG)):
            val = dP_dG[i]
            if not np.isfinite(val):
                continue
            if (val < lower) or (val > upper):
                idx_closure = int(i)
                P_closure = float(P[i])
                break

    return CastilloCycleResult(
        G=G,
        P=P,
        t_s=t_s,
        dP_dG=dP_dG,
        idx_open_start=idx_open_start,
        idx_open_end=idx_open_end,
        P_star=float(P_star),
        P_isip_theoretical=float(P_isip),
        idx_closure=idx_closure,
        P_closure=P_closure,
        r2_open=float(r2_open),
    )


# ------------------------------
# Plotting for one cycle
# ------------------------------
def plot_castillo_cycle(df_cycle: pd.DataFrame,
                        res: CastilloCycleResult,
                        title: str = ""):

    G = res.G
    P = res.P
    t_s = res.t_s
    dP_dG = res.dP_dG

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    # ------ P vs G ------
    ax1.plot(G, P, ".", label="Data")
    G_line = np.linspace(np.nanmin(G), np.nanmax(G), 200)
    P_line = res.P_star * G_line + res.P_isip_theoretical
    ax1.plot(G_line, P_line, "-", label=f"Linear fit (R²={res.r2_open:.3f})")

    # Open-fracture window
    ax1.axvspan(G[res.idx_open_start], G[res.idx_open_end],
                color="lightgrey", alpha=0.4, label="Open-fracture window")

    # Closure point
    if res.idx_closure is not None:
        ax1.axvline(G[res.idx_closure], color="r", linestyle="--")
        ax1.plot(G[res.idx_closure], P[res.idx_closure], "ro",
                 label=f"Closure ≈ {res.P_closure:.1f} bar")

    # Theoretical ISIP at G=0 (horizontal line)
    ax1.axhline(res.P_isip_theoretical, color="k", linestyle="--",
                label=f"ISIP_th ≈ {res.P_isip_theoretical:.1f} bar")

    ax1.set_xlabel("G-function")
    ax1.set_ylabel("Pressure [bar]")
    if title:
        ax1.set_title(title)
    ax1.legend(loc="best")

    # ------ dP/dG vs Δt ------
    ax2.plot(t_s, dP_dG, ".-", label="dP/dG (rolling regression)")
    if res.idx_closure is not None:
        ax2.axvline(t_s[res.idx_closure], color="r", linestyle="--",
                    label="Closure time")
    ax2.set_xlabel("Δt since shut-in [s]")
    ax2.set_ylabel("dP/dG")
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig


