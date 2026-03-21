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

# === CNN imports ===
# 1) numpy version (как у тебя сейчас)
from cnn_dpd import cnn_dpd

from cnn_dpd_torch import cnn_dpd_torch
HAS_TORCH_CNN = True


def main():
    plt.close('all')

    # -----------------------------
    # Choose method: "ls" or "lms" or "cnn"
    # -----------------------------
    method = "cnn"

    # if method == "cnn": choose implementation
    # "numpy" -> cnn_dpd from cnn_dpd.py
    # "torch" -> cnn_dpd_torch from cnn_dpd_torch.py (if exists)
    cnn_backend = "torch"  # "numpy" or "torch"

    prm = {
        'sizeSig': int(2e4),
        'txFs': 100e6,
        'sigBand': 20e6,
        'up': 4
    }

    # -----------------------------
    # CNN params
    # -----------------------------
    prm['cnn'] = {
        'memory': 5,
        'kernel': 5,
        'epochs': 1000,
        'lr': 1e-3,
        'M1': 16,
        'M2': 16,
        'filters': 8,
        'seed': 42,
        'features': 'poly',
        'print_every': 10,
        'clip': 0.0,            # numpy code likely uses this
        'weight_decay': 0.0,
        'debug_stats': False,
        'ila_iters': 10,
        'warm_start': False
    }

    prm['cnn'].update({
        'kernel': 5,
        'filters': 8,
        'M1': 16,
        'M2': 16,
        'epochs': 50,
        'lr': 5e-3,
        'print_every': 10,
        'features': 'poly',
        'grad_clip': 1.0,   # torch code might use this
        'device': 'cpu',
    })

    # keep backward compatibility: if some code expects 'clip'
    if 'grad_clip' in prm['cnn'] and ('clip' not in prm['cnn'] or prm['cnn']['clip'] == 0.0):
        prm['cnn']['clip'] = prm['cnn']['grad_clip']

    # -----------------------------
    # PA params
    # -----------------------------
    prm['pa_alpha'] = 0.8
    prm['pa_memory'] = 3
    prm['mem_decay'] = 0.7
    prm['gmp_k'] = 2
    prm['gmp_beta'] = 0.15

    # -----------------------------
    # Generate + normalize reference
    # -----------------------------
    sig = generator(prm)
    x = sig / (np.max(np.abs(sig)) + 1e-15)

    # PA modeling
    y = amp_model(prm, x)
    ampGain = np.sqrt(np.mean(np.abs(y)**2) / (np.mean(np.abs(x)**2) + 1e-15))
    #y = y / (np.max(np.abs(y)) + 1e-15)

    prm['cnn']['ampGain'] = float(ampGain)

    # -----------------------------
    # Align
    # -----------------------------
    x_al, y_al, lag = align_by_xcorr(x, y, max_lag=300)
    print(f"Alignment lag = {lag} samples. Using aligned length = {len(x_al)}")

    # -----------------------------
    # DPD
    # -----------------------------
    if method.lower() == "ls":
        orders = (1, 3, 5)
        ridge = 1e-6
        a = ls_postdistorter_coeffs(y_al, x_al, orders=orders, ridge=ridge)
        x_dpd = apply_predistorter(x_al, a, orders=orders)
        model = {'a': a, 'orders': orders, 'ridge': ridge}

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
            if not HAS_TORCH_CNN:
                raise ImportError("cnn_backend='torch', but cnn_dpd_torch.py not found/import failed.")
            x_dpd, model = cnn_dpd_torch(x_al, y_al, prm, pa_fn=lambda z: amp_model(prm, z))
        else:
            x_dpd, model = cnn_dpd(x_al, y_al, prm)

    else:
        raise ValueError('method must be "ls" or "lms" or "cnn"')

    # -----------------------------
    # (оставляю твой power check как есть)
    # -----------------------------
    p_ref = np.mean(np.abs(x_al)**2)
    p_dpd = np.mean(np.abs(x_dpd)**2) + 1e-15
    # x_dpd = x_dpd * np.sqrt(p_ref / p_dpd)

    # PA output after DPD
    y_lin = amp_model(prm, x_dpd)

    # -----------------------------
    # Metrics (NMSE to x_al)
    # -----------------------------
    nmse_before = nmse_db_gain_aligned(y_al, x_al)
    nmse_after  = nmse_db_gain_aligned(y_lin, x_al)
    print(f"Gain-aligned NMSE before DPD: {nmse_before:.2f} dB")
    print(f"Gain-aligned NMSE after  DPD: {nmse_after:.2f} dB")

    # -----------------------------
    # PSD plot (Welch)
    # -----------------------------
    fs = prm['txFs'] * prm['up']

    f, Pxx_before = signal.welch(
        y_al, fs=fs, window='hann', nperseg=4096, noverlap=2048,
        return_onesided=False, scaling='density'
    )
    _, Pxx_after = signal.welch(
        y_lin, fs=fs, window='hann', nperseg=4096, noverlap=2048,
        return_onesided=False, scaling='density'
    )

    Pxx_before = np.fft.fftshift(Pxx_before)
    Pxx_after = np.fft.fftshift(Pxx_after)
    f = np.fft.fftshift(f)

    center = len(Pxx_before) // 2
    if center > 0:
        Pxx_before[center] = Pxx_before[center - 1]
        Pxx_after[center] = Pxx_after[center - 1]

    PxxB_db = 10*np.log10(Pxx_before/(np.max(Pxx_before) + 1e-15) + 1e-15)
    PxxA_db = 10*np.log10(Pxx_after /(np.max(Pxx_after) + 1e-15) + 1e-15)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure()
    plt.plot(f / 1e6, PxxB_db, 'r', linewidth=1.5, label='До DPD')
    plt.plot(f / 1e6, PxxA_db, 'b', linewidth=1.5, label=f'После DPD')
    plt.xlabel('Частота, МГц')
    plt.ylabel('Спектральная плотность мощности, дБ')
    plt.title('Спектральная плотность мощности на выходе усилителя')
    plt.xlim([-100, 100])
    plt.ylim([-60, 0])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    fs = prm['txFs'] * prm['up']
    bw_aclr = prm['sigBand'] + 5e6
    
    fig, ax, met = plot_aclr_nr_style(
        x_before=y_al,
        x_after=y_lin,
        fs=fs,
        bw=bw_aclr,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-100, 100),
        ylim_db=(-70, 5),
        title='ACLR для сигнала на выходе усилителя'
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
    
    plt.show() 


if __name__ == "__main__":
    main()