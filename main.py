import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from generator import generator
from amp_model import amp_model

from ls_alg import align_by_xcorr, ls_postdistorter_coeffs, apply_predistorter, nmse_db, nmse_db_gain_aligned
from cnn_dpd import cnn_dpd

def main():
    plt.close('all')

    # -----------------------------
    # Choose method: "ls" or "cnn"
    # -----------------------------
    method = "ls"   

    prm = {
        'sizeSig': int(2e4),
        'txFs': 100e6,
        'sigBand': 20e6,
        'up': 4
    }

    # CNN params (используются только если method == "cnn")
    prm['cnn'] = {
        'memory': 5,
        'epochs': 500,
        'lr': 1e-2,
        'M1': 16,
        'M2': 16,
        'filters': 8,
        'seed': 42,
        'monitor': False,
        'print_every': 1
    }

    # -----------------------------
    # Generate + normalize reference
    # -----------------------------
    sig = generator(prm)
    x = sig / (np.max(np.abs(sig)) + 1e-15)

    # PA output (no DPD)
    y = amp_model(prm, x)

    # Align (полезно, особенно если up>1 и фильтры)
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

    elif method.lower() == "cnn":
        # IMPORTANT: cnn_dpd expects (clean, distorted) in the same time alignment
        # We already aligned x_al, y_al, so pass those.
        x_dpd, model = cnn_dpd(x_al, y_al, prm)
        # cnn_dpd возвращает в исходной шкале (как x_al), но у нас x_al уже нормирован.
        # Для безопасности можно ещё раз нормировать по max:
        # x_dpd = x_dpd / (np.max(np.abs(x_dpd)) + 1e-15)

    else:
        raise ValueError('method must be "ls" or "cnn"')
    
    p_ref = np.mean(np.abs(x_al)**2)
    p_dpd = np.mean(np.abs(x_dpd)**2) + 1e-15
    #x_dpd = x_dpd * np.sqrt(p_ref / p_dpd)

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

    #ref = np.max(Pxx_before) + 1e-15
    PxxB_db = 10*np.log10(Pxx_before/(np.max(Pxx_before) + 1e-15) + 1e-15)
    PxxA_db = 10*np.log10(Pxx_after /(np.max(Pxx_after) + 1e-15) + 1e-15)

    plt.figure()
    plt.plot(f / 1e6, PxxB_db, 'r', linewidth=1.5, label='Before DPD')
    plt.plot(f / 1e6, PxxA_db, 'b', linewidth=1.5, label=f'After DPD (method={method})')
    plt.xlabel('Frequency, MHz')
    plt.ylabel('PSD, dB (normalized)')
    plt.title('Power Spectral Density (PA output)')
    plt.xlim([-100, 100])
    plt.ylim([-100, 0])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()