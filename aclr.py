import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def welch_psd_db(x, fs, nperseg=4096, noverlap=2048, floor_db=-120):
    """
    Two-sided Welch PSD for complex baseband signal.
    Returns frequency in Hz and PSD in dB (absolute, not normalized to each curve separately).
    """
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling='density'
    )
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    Pxx_db = 10 * np.log10(Pxx + 1e-20)
    Pxx_db = np.maximum(Pxx_db, floor_db)
    return f, Pxx, Pxx_db


def band_power_from_psd(f, Pxx, center_hz, bw_hz):
    """
    Integrate PSD over rectangular band.
    """
    lo = center_hz - bw_hz / 2
    hi = center_hz + bw_hz / 2
    m = (f >= lo) & (f < hi)
    if np.count_nonzero(m) < 2:
        return 0.0
    return float(np.trapezoid(Pxx[m], f[m]))


def aclr(f, Pxx, bw):
    """
    3GPP-style ACLR for main channel at DC and adjacent channels centered at ±bw, ±2bw.
    Returns positive ACLR in dB:
        ACLR = 10*log10(P_main / P_adj)
    """
    P_main = band_power_from_psd(f, Pxx, 0.0, bw)

    P_adj_m1 = band_power_from_psd(f, Pxx, -bw, bw)
    P_adj_p1 = band_power_from_psd(f, Pxx, +bw, bw)

    P_adj_m2 = band_power_from_psd(f, Pxx, -2 * bw, bw)
    P_adj_p2 = band_power_from_psd(f, Pxx, +2 * bw, bw)

    def safe_aclr(p_adj):
        if p_adj <= 0 or P_main <= 0:
            return np.inf
        return 10 * np.log10(P_main / p_adj)

    return {
        "P_main": P_main,
        "P_adj_m1": P_adj_m1,
        "P_adj_p1": P_adj_p1,
        "P_adj_m2": P_adj_m2,
        "P_adj_p2": P_adj_p2,
        "aclr_m1_db": safe_aclr(P_adj_m1),
        "aclr_p1_db": safe_aclr(P_adj_p1),
        "aclr_m2_db": safe_aclr(P_adj_m2),
        "aclr_p2_db": safe_aclr(P_adj_p2),
        "leak_m1_dbc": -safe_aclr(P_adj_m1),
        "leak_p1_dbc": -safe_aclr(P_adj_p1),
        "leak_m2_dbc": -safe_aclr(P_adj_m2),
        "leak_p2_dbc": -safe_aclr(P_adj_p2),
    }


def _add_band(ax, x0_mhz, x1_mhz, label=None, alpha=0.10):
    ax.axvspan(x0_mhz, x1_mhz, alpha=alpha)
    if label is not None:
        xm = 0.5 * (x0_mhz + x1_mhz)
        ylim = ax.get_ylim()
        y = ylim[1] - 0.06 * (ylim[1] - ylim[0])
        ax.text(xm, y, label, ha='center', va='top', fontsize=10)


def plot_aclr_nr_style(
    x_before,
    x_after,
    fs,
    bw,
    nperseg=4096,
    noverlap=2048,
    xlim_mhz=None,
    ylim_db=None,
    title="Спектральная плотность мощности и полосы ACLR",
    show_second_adjacent=True,
):
    """
    MATLAB-like NR ACLR plot:
    - PSD before/after
    - main / adjacent bands highlighted
    - ACLR values annotated
    """
    f_b, Pxx_b, Pxx_b_db = welch_psd_db(x_before, fs, nperseg=nperseg, noverlap=noverlap)
    f_a, Pxx_a, Pxx_a_db = welch_psd_db(x_after,  fs, nperseg=nperseg, noverlap=noverlap)

    # Common normalization for display relative to the same reference
    ref = max(np.max(Pxx_b), np.max(Pxx_a))
    Pxx_b_rel_db = 10 * np.log10(Pxx_b / (ref + 1e-20) + 1e-20)
    Pxx_a_rel_db = 10 * np.log10(Pxx_a / (ref + 1e-20) + 1e-20)

    met_b = aclr(f_b, Pxx_b, bw)
    met_a = aclr(f_a, Pxx_a, bw)

    f_mhz = f_b / 1e6
    bw_mhz = bw / 1e6

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(10, 6))

    # Band highlighting
    if show_second_adjacent:
        _add_band(ax, -2.5 * bw_mhz, -1.5 * bw_mhz, "Adj -2", alpha=0.06)
        _add_band(ax, +1.5 * bw_mhz, +2.5 * bw_mhz, "Adj +2", alpha=0.06)

    _add_band(ax, -1.5 * bw_mhz, -0.5 * bw_mhz, "Adj -1", alpha=0.08)
    _add_band(ax, -0.5 * bw_mhz, +0.5 * bw_mhz, "Main",  alpha=0.10)
    _add_band(ax, +0.5 * bw_mhz, +1.5 * bw_mhz, "Adj +1", alpha=0.08)

    # Boundary lines
    boundaries = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] if show_second_adjacent else [-1.5, -0.5, 0.5, 1.5]
    for c in boundaries:
        ax.axvline(c * bw_mhz, color='k', linestyle='--', linewidth=0.8)

    # Curves
    ax.plot(f_mhz, Pxx_b_rel_db, linewidth=1.8, label='До DPD')
    ax.plot(f_mhz, Pxx_a_rel_db, linewidth=1.8, label='После DPD')

    ax.set_xlabel('Частота, МГц')
    ax.set_ylabel('Нормированная PSD, дБ')
    ax.set_title(title)
    ax.grid(True)

    if xlim_mhz is None:
        if show_second_adjacent:
            xlim_mhz = (-3 * bw_mhz, 3 * bw_mhz)
        else:
            xlim_mhz = (-2 * bw_mhz, 2 * bw_mhz)
    ax.set_xlim(xlim_mhz)

    if ylim_db is not None:
        ax.set_ylim(ylim_db)
    else:
        ax.set_ylim([-70, 5])

    ax.legend(loc='upper right')

    # Text box with ACLR
    text_before = (
        "До DPD\n"
        f"ACLR(-1) = {met_b['aclr_m1_db']:.2f} dB\n"
        f"ACLR(+1) = {met_b['aclr_p1_db']:.2f} dB"
    )
    text_after = (
        "После DPD\n"
        f"ACLR(-1) = {met_a['aclr_m1_db']:.2f} dB\n"
        f"ACLR(+1) = {met_a['aclr_p1_db']:.2f} dB"
    )

    if show_second_adjacent:
        text_before += (
            f"\nACLR(-2) = {met_b['aclr_m2_db']:.2f} dB"
            f"\nACLR(+2) = {met_b['aclr_p2_db']:.2f} dB"
        )
        text_after += (
            f"\nACLR(-2) = {met_a['aclr_m2_db']:.2f} dB"
            f"\nACLR(+2) = {met_a['aclr_p2_db']:.2f} dB"
        )

    ax.text(
        0.02, 0.03, text_before,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    ax.text(
        0.72, 0.03, text_after,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    fig.tight_layout()

    return fig, ax, {"before": met_b, "after": met_a}