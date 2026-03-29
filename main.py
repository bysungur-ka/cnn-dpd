import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from generator import generator
from amp_model import amp_model
from aclr import plot_aclr_nr_style

from ls_alg import (
    align_by_xcorr,
    ls_postdistorter_coeffs,
    apply_predistorter,
    nmse_db_gain_aligned
)
from lms_alg import lms_postdistorter_coeffs

from cnn_dpd import cnn_dpd
from cnn_dpd_torch import cnn_dpd_torch


def build_params():
    prm = {
        'sizeSig': int(4e4),
        'txFs': 100e6,
        'sigBand': 20e6,
        'up': 4,

        # PA params
        'pa_mode': 'iir',          # 'gmp' or 'iir'
        'pa_alpha': 0.8,
        'pa_memory': 3,
        'mem_decay': 0.7,
        'gmp_k': 2,
        'gmp_beta': 0.15,

        # IIR mode params
        'pa_b': [0.85, 0.12],
        'pa_a': [1.0, -0.55, 0.16],
        'pa_gain': 1.0,
    }

    prm['cnn'] = {
        'memory': 5,
        'kernel': 5,
        'filters': 8,
        'M1': 16,
        'epochs': 120,
        'lr': 1e-3,
        'seed': 42,
        'features': 'poly',
        'print_every': 10,
        'clip': 1.0,
        'weight_decay': 0.0,
        'ila_iters': 10,
        'warm_start': True,
        'batch_size': 4096,
        'batch_mode': 'contig',
        'residual': True,
        'power_constraint': True,
        'device': 'cpu',
    }

    return prm


def run_dpd(method, cnn_backend, x_al, y_al, prm):
    if method.lower() == "ls":
        orders = (1, 3, 5, 7)
        memory_depth = 5
        ridge = 1e-3

        a = ls_postdistorter_coeffs(
            y_al,
            x_al,
            orders=orders,
            memory_depth=memory_depth,
            ridge=ridge,
            normalize_gain=True,
        )

        x_dpd = apply_predistorter(
            x_al,
            a,
            orders=orders,
            memory_depth=memory_depth,
        )

        model = {
            'a': a,
            'orders': orders,
            'memory_depth': memory_depth,
            'ridge': ridge,
       }
        
    elif method.lower() == "lms":
        orders = (1, 3, 5)
        mu = 1e-2
        epochs_lms = 100

        a = lms_postdistorter_coeffs(
            y_al, x_al,
            orders=orders,
            mu=mu,
            epochs=epochs_lms
        )
        x_dpd = apply_predistorter(x_al, a, orders=orders)
        model = {'a': a, 'orders': orders, 'mu': mu, 'epochs': epochs_lms}

    elif method.lower() == "cnn":
        if cnn_backend == "torch":
            x_dpd, model = cnn_dpd_torch(
                x_al,
                y_al,
                prm,
                pa_fn=lambda z: amp_model(prm, z)
            )
        elif cnn_backend == "numpy":
            x_dpd, model = cnn_dpd(x_al, y_al, prm)
        else:
            raise ValueError('cnn_backend must be "torch" or "numpy"')

    else:
        raise ValueError('method must be "ls" or "lms" or "cnn"')

    return x_dpd, model


def welch_psd_db(x, fs, nperseg=4096, noverlap=2048):
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling='density'
    )

    Pxx = np.fft.fftshift(Pxx)
    f = np.fft.fftshift(f)

    center = len(Pxx) // 2
    if center > 0:
        Pxx[center] = Pxx[center - 1]

    Pxx_db = 10 * np.log10(Pxx / (np.max(Pxx) + 1e-15) + 1e-15)
    return f, Pxx_db


def plot_output_psd(y_before, y_after, prm):
    fs = prm['txFs'] * prm['up']

    f, PxxB_db = welch_psd_db(y_before, fs=fs)
    _, PxxA_db = welch_psd_db(y_after, fs=fs)

    plt.figure()
    plt.plot(f / 1e6, PxxB_db, linewidth=1.5, label='До DPD')
    plt.plot(f / 1e6, PxxA_db, linewidth=1.5, label='После DPD')
    plt.xlabel('Частота, МГц')
    plt.ylabel('Спектральная плотность мощности, дБ')
    plt.title('Спектральная плотность мощности на выходе усилителя')
    plt.xlim([-100, 100])
    plt.ylim([-50, 0])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def gain_align(y, x):
    denom = np.vdot(x, x) + 1e-15
    G = np.vdot(x, y) / denom
    y_al = y / (G + 1e-15)
    return y_al, G


def _scatter_stride(n, max_points=25000):
    return max(1, int(np.ceil(n / max_points)))


def plot_amam_ampm(x_ref, y_before, y_after):
    yb_al, _ = gain_align(y_before, x_ref)
    ya_al, _ = gain_align(y_after, x_ref)

    a_in = np.abs(x_ref)
    a_out_before = np.abs(yb_al)
    a_out_after = np.abs(ya_al)

    phi_before = np.angle(yb_al * np.conj(x_ref), deg=True)
    phi_after = np.angle(ya_al * np.conj(x_ref), deg=True)

    thr_phi = 0.05 * np.max(a_in)
    mask_phi = a_in > thr_phi

    stride_am = _scatter_stride(len(a_in), max_points=30000)
    stride_pm = _scatter_stride(np.count_nonzero(mask_phi), max_points=30000)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(
        a_in[::stride_am],
        a_out_before[::stride_am],
        s=10,
        alpha=0.45,
        label='До DPD'
    )
    ax[0].scatter(
        a_in[::stride_am],
        a_out_after[::stride_am],
        s=10,
        alpha=0.45,
        label='После DPD'
    )

    lim = max(np.max(a_in), np.max(a_out_before), np.max(a_out_after))
    ax[0].plot([0, lim], [0, lim], '--', linewidth=1.0, label='Идеальная линейность')
    ax[0].set_xlabel('Амплитуда входного сигнала')
    ax[0].set_ylabel('Амплитуда выходного сигнала')
    ax[0].set_title('AM/AM характеристика')
    ax[0].grid(True)
    ax[0].legend()

    a_in_phi = a_in[mask_phi][::stride_pm]
    phi_b = phi_before[mask_phi][::stride_pm]
    phi_a = phi_after[mask_phi][::stride_pm]

    ax[1].scatter(
        a_in_phi,
        phi_b,
        s=10,
        alpha=0.45,
        label='До DPD'
    )
    ax[1].scatter(
        a_in_phi,
        phi_a,
        s=10,
        alpha=0.45,
        label='После DPD'
    )
    ax[1].axhline(0.0, linestyle='--', linewidth=1.0, label='Идеальная линейность')
    ax[1].set_xlabel('Амплитуда входного сигнала')
    ax[1].set_ylabel('Фазовая ошибка, градусы')
    ax[1].set_title('AM/PM характеристика')
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle('Амплитудно-амплитудная и амплитудно-фазовая характеристики каскада')
    fig.tight_layout()


def plot_gain_vs_input(x_ref, y_before, y_after):
    yb_al, _ = gain_align(y_before, x_ref)
    ya_al, _ = gain_align(y_after, x_ref)

    eps = 1e-15

    pin_db = 20 * np.log10(np.abs(x_ref) + eps)
    pout_before_db = 20 * np.log10(np.abs(yb_al) + eps)
    pout_after_db = 20 * np.log10(np.abs(ya_al) + eps)

    gain_before_db = pout_before_db - pin_db
    gain_after_db = pout_after_db - pin_db

    thr = 0.02 * np.max(np.abs(x_ref))
    mask = np.abs(x_ref) > thr

    pin_db = pin_db[mask]
    gain_before_db = gain_before_db[mask]
    gain_after_db = gain_after_db[mask]

    stride = _scatter_stride(len(pin_db), max_points=30000)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        pin_db[::stride],
        gain_before_db[::stride],
        s=10,
        alpha=0.45,
        label='До DPD'
    )
    plt.scatter(
        pin_db[::stride],
        gain_after_db[::stride],
        s=10,
        alpha=0.45,
        label='После DPD'
    )
    plt.axhline(0.0, linestyle='--', linewidth=1.0, label='Идеально постоянное усиление')
    plt.xlabel('Уровень входного сигнала, дБ')
    plt.ylabel('Коэффициент усиления, дБ')
    plt.title('Gain vs Input Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_ila_history(model):
    if not isinstance(model, dict):
        return

    nmse_after = np.asarray(model.get('nmse_after_hist_db', []), dtype=float)
    if nmse_after.size == 0:
        return

    plt.figure()
    plt.plot(
        np.arange(1, len(nmse_after) + 1),
        nmse_after,
        marker='o',
        linewidth=1.8,
        label='NMSE'
    )
    plt.xlabel('Номер итерации ILA')
    plt.ylabel('NMSE, дБ')
    plt.title('Сходимость ILA по системной метрике NMSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    plt.close('all')
    plt.rcParams['font.family'] = 'DejaVu Sans'

    method = "ls"           # "ls", "lms", "cnn"
    cnn_backend = "torch"    # "torch" or "numpy"

    prm = build_params()

    sig = generator(prm)
    x = sig / (np.max(np.abs(sig)) + 1e-15)

    y = amp_model(prm, x)

    x_al, y_al, lag = align_by_xcorr(x, y, max_lag=300)
    print(f"Alignment lag = {lag} samples. Using aligned length = {len(x_al)}")

    x_dpd, model = run_dpd(method, cnn_backend, x_al, y_al, prm)

    # -----------------------------
    # Normalize DPD drive level before PA
    # -----------------------------
    p_ref = np.mean(np.abs(x_al)**2) + 1e-15
    p_dpd = np.mean(np.abs(x_dpd)**2) + 1e-15
    x_dpd = x_dpd * np.sqrt(p_ref / p_dpd)

    print(f"Input RMS power before DPD drive norm: {10*np.log10(p_dpd):.2f} dB")
    print(f"Reference RMS power: {10*np.log10(p_ref):.2f} dB")

    y_lin = amp_model(prm, x_dpd)

    nmse_before = nmse_db_gain_aligned(y_al, x_al)
    nmse_after = nmse_db_gain_aligned(y_lin, x_al)

    print(f"Gain-aligned NMSE before DPD: {nmse_before:.2f} dB")
    print(f"Gain-aligned NMSE after  DPD: {nmse_after:.2f} dB")

    plot_output_psd(y_al, y_lin, prm)

    fs = prm['txFs'] * prm['up']
    bw_aclr = prm['sigBand'] + 5e6

    # Общая нормировка: честное сравнение по уровню
    fig, ax, met = plot_aclr_nr_style(
        x_before=y_al,
        x_after=y_lin,
        fs=fs,
        bw=bw_aclr,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-100, 100),
        ylim_db=(-70, 5),
        title='ACLR для сигнала на выходе усилителя (общая нормировка)',
        common_ref=True
    )

    # Индивидуальная нормировка: сравнение формы спектра
    fig2, ax2, met2 = plot_aclr_nr_style(
        x_before=y_al,
        x_after=y_lin,
        fs=fs,
        bw=bw_aclr,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-100, 100),
        ylim_db=(-70, 5),
        title='ACLR для сигнала на выходе усилителя (индивидуальная нормировка)',
        common_ref=False
    )

    print("До DPD:")
    print(f"  ACLR(-1) = {met['before']['aclr_m1_db']:.2f} dB")
    print(f"  ACLR(+1) = {met['before']['aclr_p1_db']:.2f} dB")
    print(f"  Leakage(-1) = {met['before']['leak_m1_dbc']:.2f} dBc")
    print(f"  Leakage(+1) = {met['before']['leak_p1_dbc']:.2f} dBc")

    print("После DPD:")
    print(f"  ACLR(-1) = {met['after']['aclr_m1_db']:.2f} dB")
    print(f"  ACLR(+1) = {met['after']['aclr_p1_db']:.2f} dB")
    print(f"  Leakage(-1) = {met['after']['leak_m1_dbc']:.2f} dBc")
    print(f"  Leakage(+1) = {met['after']['leak_p1_dbc']:.2f} dBc")

    plot_amam_ampm(x_al, y_al, y_lin)
    plot_gain_vs_input(x_al, y_al, y_lin)
    plot_ila_history(model)

    plt.show()


if __name__ == "__main__":
    main()