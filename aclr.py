import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle


def _welch_psd(x, fs, nperseg=4096, noverlap=2048):
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

    center = len(Pxx) // 2
    if center > 0:
        Pxx[center] = Pxx[center - 1]

    return f, Pxx


def _band_power(f, Pxx, f1, f2):
    mask = (f >= f1) & (f < f2)
    if not np.any(mask):
        return 1e-20
    return np.trapezoid(Pxx[mask], f[mask]) + 1e-20


def plot_aclr_nr_style(
    x_before,
    x_after,
    fs,
    bw,
    nperseg=4096,
    noverlap=2048,
    xlim_mhz=None,
    ylim_db=None,
    title='Спектральная плотность мощности и полосы ACLR',
    show_second_adjacent=True,
    common_ref=True,
):
    """
    NR-style ACLR plot.

    Parameters
    ----------
    x_before, x_after : complex ndarray
        Signals before and after DPD
    fs : float
        Sampling frequency
    bw : float
        Main channel bandwidth
    common_ref : bool
        True  -> both spectra normalized to one common peak
        False -> each spectrum normalized to its own peak
    """

    x_before = np.asarray(x_before, dtype=np.complex128)
    x_after = np.asarray(x_after, dtype=np.complex128)

    f, Pxx_b = _welch_psd(x_before, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxx_a = _welch_psd(x_after, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Main and adjacent channels
    main = (-bw / 2, bw / 2)
    adj_m1 = (-3 * bw / 2, -bw / 2)
    adj_p1 = (bw / 2, 3 * bw / 2)

    adj_m2 = (-5 * bw / 2, -3 * bw / 2)
    adj_p2 = (3 * bw / 2, 5 * bw / 2)

    # Powers before
    Pm_b = _band_power(f, Pxx_b, *main)
    Padj_m1_b = _band_power(f, Pxx_b, *adj_m1)
    Padj_p1_b = _band_power(f, Pxx_b, *adj_p1)
    Padj_m2_b = _band_power(f, Pxx_b, *adj_m2)
    Padj_p2_b = _band_power(f, Pxx_b, *adj_p2)

    # Powers after
    Pm_a = _band_power(f, Pxx_a, *main)
    Padj_m1_a = _band_power(f, Pxx_a, *adj_m1)
    Padj_p1_a = _band_power(f, Pxx_a, *adj_p1)
    Padj_m2_a = _band_power(f, Pxx_a, *adj_m2)
    Padj_p2_a = _band_power(f, Pxx_a, *adj_p2)

    aclr_m1_b = 10 * np.log10(Pm_b / Padj_m1_b)
    aclr_p1_b = 10 * np.log10(Pm_b / Padj_p1_b)
    aclr_m2_b = 10 * np.log10(Pm_b / Padj_m2_b)
    aclr_p2_b = 10 * np.log10(Pm_b / Padj_p2_b)

    aclr_m1_a = 10 * np.log10(Pm_a / Padj_m1_a)
    aclr_p1_a = 10 * np.log10(Pm_a / Padj_p1_a)
    aclr_m2_a = 10 * np.log10(Pm_a / Padj_m2_a)
    aclr_p2_a = 10 * np.log10(Pm_a / Padj_p2_a)

    leak_m1_b = 10 * np.log10(Padj_m1_b / Pm_b)
    leak_p1_b = 10 * np.log10(Padj_p1_b / Pm_b)
    leak_m2_b = 10 * np.log10(Padj_m2_b / Pm_b)
    leak_p2_b = 10 * np.log10(Padj_p2_b / Pm_b)

    leak_m1_a = 10 * np.log10(Padj_m1_a / Pm_a)
    leak_p1_a = 10 * np.log10(Padj_p1_a / Pm_a)
    leak_m2_a = 10 * np.log10(Padj_m2_a / Pm_a)
    leak_p2_a = 10 * np.log10(Padj_p2_a / Pm_a)

    # Normalize PSD for plotting
    if common_ref:
        ref = max(np.max(Pxx_b), np.max(Pxx_a))
        Pxx_b_rel_db = 10 * np.log10(Pxx_b / (ref + 1e-20) + 1e-20)
        Pxx_a_rel_db = 10 * np.log10(Pxx_a / (ref + 1e-20) + 1e-20)
    else:
        Pxx_b_rel_db = 10 * np.log10(Pxx_b / (np.max(Pxx_b) + 1e-20) + 1e-20)
        Pxx_a_rel_db = 10 * np.log10(Pxx_a / (np.max(Pxx_a) + 1e-20) + 1e-20)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(f / 1e6, Pxx_b_rel_db, linewidth=1.5, label='До DPD')
    ax.plot(f / 1e6, Pxx_a_rel_db, linewidth=1.5, label='После DPD')

    def add_band(band, label=None, alpha=0.08):
        x0 = band[0] / 1e6
        w = (band[1] - band[0]) / 1e6
        y0, y1 = ax.get_ylim()
        rect = Rectangle((x0, y0), w, y1 - y0, fill=True, alpha=alpha)
        ax.add_patch(rect)
        if label is not None:
            ax.text(x0 + w / 2, y1 - 3, label, ha='center', va='top', fontsize=9)

    if ylim_db is not None:
        ax.set_ylim(ylim_db)

    add_band(main, 'Main')
    add_band(adj_m1, 'Adj -1')
    add_band(adj_p1, 'Adj +1')

    if show_second_adjacent:
        add_band(adj_m2, 'Adj -2', alpha=0.05)
        add_band(adj_p2, 'Adj +2', alpha=0.05)

    ax.set_xlabel('Частота, МГц')
    ax.set_ylabel('Нормированная PSD, дБ')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if xlim_mhz is not None:
        ax.set_xlim(xlim_mhz)

    metrics = {
        'before': {
            'aclr_m1_db': aclr_m1_b,
            'aclr_p1_db': aclr_p1_b,
            'aclr_m2_db': aclr_m2_b,
            'aclr_p2_db': aclr_p2_b,
            'leak_m1_dbc': leak_m1_b,
            'leak_p1_dbc': leak_p1_b,
            'leak_m2_dbc': leak_m2_b,
            'leak_p2_dbc': leak_p2_b,
        },
        'after': {
            'aclr_m1_db': aclr_m1_a,
            'aclr_p1_db': aclr_p1_a,
            'aclr_m2_db': aclr_m2_a,
            'aclr_p2_db': aclr_p2_a,
            'leak_m1_dbc': leak_m1_a,
            'leak_p1_dbc': leak_p1_a,
            'leak_m2_dbc': leak_m2_a,
            'leak_p2_dbc': leak_p2_a,
        }
    }

    fig.tight_layout()
    return fig, ax, metrics