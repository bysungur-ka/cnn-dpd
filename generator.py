import numpy as np
from scipy.signal import remez, lfilter, firwin, upfirdn, kaiserord

def generator(prm):
    """
    Генератор комплексного базового сигнала:
    1) комплексный белый шум (I/Q симметрично)
    2) полосовой (в базе) FIR lowpass через remez (в нормированных частотах 0..0.5)
    3) опциональный апсемплинг up через Kaiser FIR + upfirdn
    """
    txSigBw = 16
    margFs = 0.7
    maxAbs = int(round(2 ** (txSigBw - 1) * margFs))

    sizeSig = int(prm["sizeSig"])
    txFs = float(prm["txFs"])
    sigBand = float(prm["sigBand"])
    up = int(prm["up"])

    # -----------------------------
    # 1) Complex white noise (I/Q balanced)
    # -----------------------------
    y = ((np.random.rand(sizeSig) * 2 - 1) + 1j * (np.random.rand(sizeSig) * 2 - 1)) * maxAbs
    y = y.astype(np.complex128)

    # -----------------------------
    # 2) Baseband lowpass FIR (remez)
    # -----------------------------
    inBbFirSlopeMarg = 0.2
    Fpass = sigBand / txFs                 # normalized to Fs
    Fstop = Fpass * (1 + inBbFirSlopeMarg)

    # remez expects bands in [0, 0.5] when fs=1.0 (Nyquist=0.5)
    bands = [0.0, Fpass, Fstop, 0.5]
    desired = [1.0, 0.0]

    numtaps_bb = 129  # лучше нечётное для линейной фазы и целой задержки
    b = remez(numtaps_bb, bands, desired, fs=1.0)

    y_filt = lfilter(b, 1.0, y)

    # Компенсация групповой задержки (линейно-фазовый FIR)
    delay_bb = (len(b) - 1) // 2
    y_filt = y_filt[delay_bb:]  # сдвигаем вперёд
    # Чтобы длина была предсказуемой: подрежем хвост под исходную длину
    y_filt = y_filt[:sizeSig - delay_bb]

    # -----------------------------
    # 3) Interpolation (upfirdn) + delay compensation
    # -----------------------------
    if up > 1:
        atten = 90.0
        width = 0.1 / up  # нормированная ширина перехода
        N, beta = kaiserord(atten, width)
        N = int(np.ceil(N / up) * up)  # кратно up (polyphase-friendly)
        if N % 2 == 0:
            N += 1  # нечётная длина -> целая групповая задержка

        h = firwin(numtaps=N, cutoff=1.0 / up, window=("kaiser", beta))
        y_up = upfirdn(h, y_filt, up=up)

        delay_up = (len(h) - 1) // 2
        # upfirdn даёт задержку в выходных отсчётах (после up)
        y_up = y_up[delay_up:-delay_up] if delay_up > 0 else y_up
        return y_up.astype(np.complex128)

    return y_filt.astype(np.complex128)