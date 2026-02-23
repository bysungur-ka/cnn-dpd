import numpy as np
from scipy.signal import remez, lfilter
from scipy.signal import firwin, upfirdn, kaiserord

def generator(prm):
    
    txSigBw = 16
    margFs  = 0.7
    maxAbs = round(2**(txSigBw - 1) * margFs)

    sizeSig = prm["sizeSig"]
    txFs = prm["txFs"]
    sigBand = prm["sigBand"]
    up = prm["up"]

    # Complex white noise
    y = ((np.random.rand(sizeSig) * 2 - 1) + 1j*(np.random.rand(sizeSig) * 2 - 1))*maxAbs

    # FIR filter
    inBbFirSlopeMarg = 0.2
    Fpass = sigBand / txFs
    Fstop = Fpass * (1 + inBbFirSlopeMarg)

    bands = [0, Fpass, Fstop, 0.5]
    desired = [1, 0]

    N = 128  # фиксируем порядок
    b = remez(N, bands, desired)

    y_filt = lfilter(b, 1, y)
    
    y_filt = np.round(y_filt)

    # Interpolation

    atten = 90  # dB подавления образов
    width = 0.1 / up  # ширина переходной полосы (норм.)
    
    N, beta = kaiserord(atten, width)

    # делаем длину кратной up (лучше для polyphase)
    N = int(np.ceil(N / up) * up)

    h = firwin(
    numtaps=N,
    cutoff=1 / up,
    window=("kaiser", beta)
    )

    y_up = upfirdn(h, y_filt, up)
    
    # компенсируем групповую задержку
    delay = (len(h) - 1) // 2
    y_up = y_up[delay : -delay]

    return y_up
