import numpy as np
from scipy.signal import remez, lfilter, firwin, upfirdn, kaiserord, resample_poly


def _trim_or_pad(x: np.ndarray, target_len: int):
    x = np.asarray(x, dtype=np.complex128)
    if len(x) >= target_len:
        return x[:target_len]
    y = np.zeros(target_len, dtype=np.complex128)
    y[:len(x)] = x
    return y


def _generate_square_qam(n: int, qam_order: int, rng: np.random.Generator):
    """
    Генерация square M-QAM, например 16/64/256/1024-QAM.
    """
    m_side = int(np.sqrt(qam_order))
    if m_side * m_side != qam_order or (m_side % 2) != 0:
        raise ValueError("qam_order must be a square even-order QAM, e.g. 16, 64, 256, 1024")

    levels = np.arange(-(m_side - 1), m_side, 2)
    real = rng.choice(levels, n)
    imag = rng.choice(levels, n)
    s = real + 1j * imag
    return s / np.sqrt(np.mean(np.abs(s) ** 2) + 1e-15)


def _generate_noise_mode(prm, rng: np.random.Generator):
    """
    Старый режим:
      1) комплексный белый шум
      2) baseband lowpass FIR
      3) upsampling
    """
    txSigBw = 16
    margFs = 0.7
    maxAbs = int(round(2 ** (txSigBw - 1) * margFs))

    sizeSig = int(prm["sizeSig"])
    txFs = float(prm["txFs"])
    sigBand = float(prm["sigBand"])
    up = int(prm["up"])

    # 1) complex white noise
    y = ((rng.random(sizeSig) * 2 - 1) + 1j * (rng.random(sizeSig) * 2 - 1)) * maxAbs
    y = y.astype(np.complex128)

    # 2) baseband lowpass FIR
    inBbFirSlopeMarg = 0.2
    Fpass = sigBand / (2 * txFs)
    Fstop = Fpass * (1 + inBbFirSlopeMarg)

    bands = [0.0, Fpass, Fstop, 0.5]
    desired = [1.0, 0.0]
    numtaps_bb = 129
    b = remez(numtaps_bb, bands, desired, fs=1.0)

    y_filt = lfilter(b, 1.0, y)
    delay_bb = (len(b) - 1) // 2
    y_filt = y_filt[delay_bb:]
    y_filt = _trim_or_pad(y_filt, sizeSig)

    # 3) interpolation
    if up > 1:
        atten = 90.0
        width = 0.1 / up
        N, beta = kaiserord(atten, width)
        N = int(np.ceil(N / up) * up)
        if N % 2 == 0:
            N += 1

        h = firwin(numtaps=N, cutoff=1.0 / up, window=("kaiser", beta))
        y_up = upfirdn(h, y_filt, up=up)

        delay_up = (len(h) - 1) // 2
        y_up = y_up[delay_up:-delay_up] if delay_up > 0 else y_up

        return _trim_or_pad(y_up, sizeSig * up).astype(np.complex128)

    return y_filt.astype(np.complex128)


def _generate_ofdm_mode(prm, rng: np.random.Generator):
    """
    OFDM-like режим:
      1) генерация QAM-символов
      2) размещение в центре спектра
      3) IFFT + CP
      4) FIR-фильтрация
      5) upsampling
      6) post-interp filtering
    """
    sizeSig = int(prm["sizeSig"])
    txFs = float(prm["txFs"])
    sigBand = float(prm["sigBand"])
    up = int(prm["up"])

    # OFDM params
    N_fft = int(prm.get("ofdm_nfft", 1024))
    scs = float(prm.get("ofdm_scs", 30e3))
    n_rb = int(prm.get("ofdm_n_rb", 51))
    cp_len = int(prm.get("ofdm_cp_len", 72))
    qam_order = int(prm.get("ofdm_qam_order", 1024))

    n_subcarriers = n_rb * 12
    if n_subcarriers >= N_fft:
        raise ValueError("n_rb*12 must be smaller than N_fft")

    fs_ofdm = N_fft * scs
    if abs(fs_ofdm - txFs) > 1e-6 * max(fs_ofdm, txFs):
        raise ValueError(
            f"OFDM mode expects txFs = N_fft * scs. "
            f"Now got txFs={txFs:.3f}, but N_fft*scs={fs_ofdm:.3f}."
        )

    symbol_len = N_fft + cp_len

    # генерируем немного с запасом на фильтры
    margin = int(prm.get("ofdm_margin_samples", 4 * symbol_len))
    n_ofdm_symbols = int(np.ceil((sizeSig + margin) / symbol_len))

    ofdm_symbols = []
    center = N_fft // 2
    half = n_subcarriers // 2

    for _ in range(n_ofdm_symbols):
        freq_domain = np.zeros(N_fft, dtype=np.complex128)
        qam_syms = _generate_square_qam(n_subcarriers, qam_order, rng)
        freq_domain[center - half:center + half] = qam_syms

        time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain)) * np.sqrt(N_fft)
        cp = time_domain[-cp_len:]
        ofdm_with_cp = np.concatenate([cp, time_domain])
        ofdm_symbols.append(ofdm_with_cp)

    tx_signal = np.concatenate(ofdm_symbols)

    # ---------- baseband FIR ----------
    passband = float(prm.get("ofdm_passband", 0.45 * sigBand))
    stopband = float(prm.get("ofdm_stopband", 0.525 * sigBand))
    atten_db = float(prm.get("ofdm_filter_atten_db", 70.0))
    kaiser_beta = float(prm.get("ofdm_kaiser_beta", 10.0))

    if not (0 < passband < stopband < txFs / 2):
        raise ValueError("Need 0 < passband < stopband < txFs/2 for OFDM FIR filter")

    transition_width = stopband - passband
    delta_omega = 2 * np.pi * transition_width / txFs
    num_taps1 = int(np.ceil((atten_db - 8) / (2.285 * delta_omega))) | 1
    num_taps1 = max(num_taps1, 65)

    h1 = firwin(
        num_taps1,
        cutoff=passband,
        window=("kaiser", kaiser_beta),
        fs=txFs
    )

    tx_filtered = lfilter(h1, 1.0, tx_signal)
    delay1 = len(h1) // 2
    tx_filtered = tx_filtered[delay1:]
    tx_filtered = _trim_or_pad(tx_filtered, sizeSig)

    # ---------- interpolation ----------
    if up > 1:
        tx_upsampled = resample_poly(tx_filtered, up=up, down=1)
        fs_up = txFs * up

        cutoff2 = txFs / 2
        num_taps2 = int(prm.get("ofdm_post_num_taps", 201))
        h2 = firwin(num_taps2, cutoff2, window="hamming", fs=fs_up)

        tx_final = lfilter(h2, 1.0, tx_upsampled)
        delay2 = len(h2) // 2
        tx_final = tx_final[delay2:]
        tx_final = _trim_or_pad(tx_final, sizeSig * up)
        return tx_final.astype(np.complex128)

    return tx_filtered.astype(np.complex128)


def generator(prm):
    """
    Генератор комплексного сигнала.

    prm:
      signal_mode : 'noise' (default) or 'ofdm'

    Общие параметры:
      sizeSig, txFs, sigBand, up

    Для OFDM:
      ofdm_nfft
      ofdm_scs
      ofdm_n_rb
      ofdm_cp_len
      ofdm_qam_order
      ofdm_passband
      ofdm_stopband
      ofdm_filter_atten_db
      ofdm_post_num_taps
      seed (optional)
    """
    mode = str(prm.get("signal_mode", "noise")).lower()
    seed = prm.get("seed", None)
    rng = np.random.default_rng(seed)

    if mode == "noise":
        return _generate_noise_mode(prm, rng)

    if mode == "ofdm":
        return _generate_ofdm_mode(prm, rng)

    raise ValueError("signal_mode must be 'noise' or 'ofdm'")