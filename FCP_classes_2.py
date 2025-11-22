from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

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
    P_anchor: float             # anchor P (lowest P)
    Y_anchor: float             # anchor Y
    Y_break: float              # Y at P_break
    R2_all: float               # R² over all used points
    R2_left: float              # R² left of P_break
    R2_right: float             # R² right of P_break


class HaimsonBilinearPicker:
    """Haimson-style bilinear FCP picker on Y = -dP/dt vs P."""

    def __init__(self, min_seg_size: int = 5, skip_high_n: int = 3) -> None:
        self.min_seg_size = min_seg_size
        self.skip_high_n = skip_high_n

    @staticmethod
    def bilinear_model_anchored(
        P: np.ndarray,
        m1: float,
        m2: float,
        P_break: float,
        P_anchor: float,
        Y_anchor: float,
    ) -> np.ndarray:
        """Continuous bilinear model with left segment anchored at (P_anchor, Y_anchor)."""
        P = np.asarray(P, dtype=float)

        Y_break = Y_anchor - m1 * (P_anchor - P_break)

        Y = np.empty_like(P)
        mask_left = P <= P_break
        mask_right = ~mask_left

        Y[mask_left] = Y_break + m1 * (P[mask_left] - P_break)
        Y[mask_right] = Y_break + m2 * (P[mask_right] - P_break)

        return Y

    def fit_PY(self, P: np.ndarray, Y: np.ndarray) -> Optional[HaimsonResult]:
        """Perform the anchored Haimson bilinear fit on Y vs P."""
        P = np.asarray(P, dtype=float)
        Y = np.asarray(Y, dtype=float)

        mask = np.isfinite(P) & np.isfinite(Y)
        P = P[mask]
        Y = Y[mask]

        if P.size < 2 * self.min_seg_size + 2:
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

        # anchor point: lowest pressure datapoint (after skipping high points)
        P_anchor = P_sorted[0]
        Y_anchor = Y_sorted[0]

        # admissible range for breakpoint
        P_min = P_sorted[self.min_seg_size]
        P_max = P_sorted[n - self.min_seg_size - 1]
        if P_min >= P_max:
            return None
        P_break_init = 0.5 * (P_min + P_max)

        # initial slopes from simple two-segment linear regressions
        mid = n // 2
        P_left = P_sorted[:mid]
        Y_left = Y_sorted[:mid]
        A_left = np.vstack([P_left, np.ones_like(P_left)]).T
        m1_init, _ = np.linalg.lstsq(A_left, Y_left, rcond=None)[0]

        P_right = P_sorted[mid:]
        Y_right = Y_sorted[mid:]
        A_right = np.vstack([P_right, np.ones_like(P_right)]).T
        m2_init, _ = np.linalg.lstsq(A_right, Y_right, rcond=None)[0]

        p0 = [m1_init, m2_init, P_break_init]

        def model_wrapper(P_in, m1, m2, P_break):
            return self.bilinear_model_anchored(
                P_in, m1, m2, P_break, P_anchor, Y_anchor
            )

        lower = [-np.inf, -np.inf, P_min]
        upper = [np.inf, np.inf, P_max]

        try:
            popt, _ = curve_fit(
                model_wrapper,
                P_sorted,
                Y_sorted,
                p0=p0,
                bounds=(lower, upper),
                maxfev=10000,
            )
        except Exception:
            return None

        m1, m2, P_break = popt
        Y_fit_all = model_wrapper(P_sorted_all, m1, m2, P_break)

        Y_break = Y_anchor - m1 * (P_anchor - P_break)

        def calc_R2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
            ss_res = np.sum((y_obs - y_pred) ** 2)
            ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # R²s computed on used points only
        Y_fit_used = model_wrapper(P_sorted, m1, m2, P_break)
        R2_all = calc_R2(Y_sorted, Y_fit_used)

        mask_left = P_sorted <= P_break
        mask_right = ~mask_left

        if mask_left.sum() >= 2:
            R2_left = calc_R2(Y_sorted[mask_left], Y_fit_used[mask_left])
        else:
            R2_left = float("nan")

        if mask_right.sum() >= 2:
            R2_right = calc_R2(Y_sorted[mask_right], Y_fit_used[mask_right])
        else:
            R2_right = float("nan")

        return HaimsonResult(
            P_sorted=P_sorted_all,
            Y_sorted=Y_sorted_all,
            Y_fit=Y_fit_all,
            m1=float(m1),
            m2=float(m2),
            P_break=float(P_break),
            P_anchor=float(P_anchor),
            Y_anchor=float(Y_anchor),
            Y_break=float(Y_break),
            R2_all=float(R2_all),
            R2_left=float(R2_left),
            R2_right=float(R2_right),
        )


# -----------------------------
# System stiffness picker
# -----------------------------
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
    dP_dG: np.ndarray  # stored NEGATIVE derivative used for picking


def system_stiffness_pick(
    G: np.ndarray,
    P: np.ndarray,
    t: Optional[np.ndarray] = None,
    smooth: bool = False,
    smooth_window: int = 5,
) -> StiffnessPick:
    """System stiffness pick based on the minimum of -dP/dG."""
    G = np.asarray(G, dtype=float)
    P = np.asarray(P, dtype=float)

    if t is not None:
        t = np.asarray(t, dtype=float)

    mask = np.isfinite(G) & np.isfinite(P)
    if t is not None:
        mask &= np.isfinite(t)

    G = G[mask]
    P = P[mask]
    if t is not None:
        t = t[mask]

    if G.size < 3:
        raise ValueError("Not enough points for system stiffness pick.")

    dP_dG_raw = np.gradient(P, G)
    neg_dP_dG = -dP_dG_raw

    if smooth and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window) / smooth_window
        pad = smooth_window // 2
        neg_dP_dG = np.convolve(
            np.pad(neg_dP_dG, pad_width=pad, mode="edge"),
            kernel,
            mode="valid",
        )

    idx_min = int(np.nanargmin(neg_dP_dG))
    dP_dG_min = float(neg_dP_dG[idx_min])

    dP_dG_target = 1.10 * dP_dG_min

    idx_target = None
    for i in range(idx_min + 1, len(neg_dP_dG)):
        if neg_dP_dG[i] >= dP_dG_target:
            idx_target = int(i)
            break

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
        dP_dG=neg_dP_dG,
    )


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
def castillo_rolling_dP_dG(
    G: np.ndarray,
    P: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """Rolling dP/dG using local linear regression (Castillo 1987 style)."""
    G = np.asarray(G, dtype=float)
    P = np.asarray(P, dtype=float)

    mask = np.isfinite(G) & np.isfinite(P)
    G = G[mask]
    P = P[mask]

    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2

    d = np.full_like(G, np.nan)

    for i in range(half, len(G) - half):
        g_win = G[i - half : i + half + 1]
        p_win = P[i - half : i + half + 1]
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
    open_frac_window=(0.1, 0.5),
    deriv_window: int = 7,
    deriv_tol_rel: float = 0.10,
    t_min_for_linear: float = 0.0,
) -> CastilloCycleResult:
    """Castillo (1987) G-function analysis on a single cycle."""
    t_s = df_cycle[col_t].to_numpy(dtype=float)
    G = df_cycle[col_G].to_numpy(dtype=float)
    P = df_cycle[col_P].to_numpy(dtype=float)

    mask = np.isfinite(t_s) & np.isfinite(G) & np.isfinite(P)
    t_s = t_s[mask]
    G = G[mask]
    P = P[mask]

    if len(G) < 10:
        raise ValueError("Not enough points in cycle for Castillo analysis.")

    dP_dG = castillo_rolling_dP_dG(G, P, window=deriv_window)

    # restrict to times >= t_min_for_linear
    mask_time = t_s >= t_min_for_linear
    t_lin = t_s[mask_time]
    G_lin = G[mask_time]
    P_lin = P[mask_time]
    dP_dG_lin = dP_dG[mask_time]

    if len(G_lin) < 5:
        raise ValueError("Not enough points after t_min_for_linear.")

    gmin = np.nanmin(G_lin)
    gmax = np.nanmax(G_lin)
    g_start_frac, g_end_frac = open_frac_window
    G_start = gmin + g_start_frac * (gmax - gmin)
    G_end = gmin + g_end_frac * (gmax - gmin)

    mask_open = (G_lin >= G_start) & (G_lin <= G_end)
    G_open = G_lin[mask_open]
    P_open = P_lin[mask_open]

    if G_open.size < 3:
        raise ValueError("Open-fracture window too small for regression.")

    A_open = np.vstack([G_open, np.ones_like(G_open)]).T
    m_open, b_open = np.linalg.lstsq(A_open, P_open, rcond=None)[0]

    P_star = float(m_open)
    P_isip_theoretical = float(b_open)

    P_pred_open = m_open * G_open + b_open
    ss_res = np.sum((P_open - P_pred_open) ** 2)
    ss_tot = np.sum((P_open - np.mean(P_open)) ** 2)
    r2_open = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    idx_closure = None
    for i in range(len(G_lin)):
        if not np.isfinite(dP_dG_lin[i]):
            continue
        if dP_dG_lin[i] < P_star * (1.0 - deriv_tol_rel):
            idx_closure = i
            break

    P_closure = float(P_lin[idx_closure]) if idx_closure is not None else None

    i0 = np.where(mask_time)[0][0]
    open_idx_full = np.where(mask_time)[0][mask_open]
    idx_open_start = int(open_idx_full[0])
    idx_open_end = int(open_idx_full[-1])

    if idx_closure is not None:
        idx_closure_full = int(np.where(mask_time)[0][idx_closure])
    else:
        idx_closure_full = None

    return CastilloCycleResult(
        G=G,
        P=P,
        t_s=t_s,
        dP_dG=dP_dG,
        idx_open_start=idx_open_start,
        idx_open_end=idx_open_end,
        P_star=P_star,
        P_isip_theoretical=P_isip_theoretical,
        idx_closure=idx_closure_full,
        P_closure=P_closure,
        r2_open=float(r2_open),
    )


# ------------------------------
# Barree-style tangent picker
# ------------------------------
@dataclass
class BarreeResult:
    x: np.ndarray
    P: np.ndarray
    dP_dlogx: np.ndarray
    idx_closure: Optional[int]
    P_closure: Optional[float]


def barree_pick(
    x: np.ndarray,
    P: np.ndarray,
    smooth_window: int = 5,
) -> BarreeResult:
    """Very simple Barree-style picker on a semilog derivative curve."""
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)

    mask = np.isfinite(x) & np.isfinite(P) & (x > 0)
    x = x[mask]
    P = P[mask]

    if x.size < 5:
        return BarreeResult(
            x=x,
            P=P,
            dP_dlogx=np.full_like(x, np.nan),
            idx_closure=None,
            P_closure=None,
        )

    dPdx = np.gradient(P, x)
    dP_dlogx = x * dPdx

    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window) / smooth_window
        dP_dlogx = np.convolve(dP_dlogx, kernel, mode="same")

    n = x.size
    n_tail = max(int(0.2 * n), 5)
    idx_tail_start = n - n_tail
    xt = x[idx_tail_start:]
    yt = dP_dlogx[idx_tail_start:]
    A = np.vstack([xt, np.ones_like(xt)]).T
    m, b = np.linalg.lstsq(A, yt, rcond=None)[0]
    trend = m * x + b

    diff = dP_dlogx - trend
    idx_closure = int(np.argmax(diff))
    P_closure = float(P[idx_closure])

    return BarreeResult(
        x=x,
        P=P,
        dP_dlogx=dP_dlogx,
        idx_closure=idx_closure,
        P_closure=P_closure,
    )