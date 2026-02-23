import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Import required modules (user-provided)
from generator import generator
from amp_model import amp_model
from cnn_dpd import cnn_dpd

def plot_spectrum(sig, fs):
    sig = np.asarray(sig)

    f, Pxx = signal.welch(
        sig,
        fs=fs,
        window='hann',
        nperseg=4096,
        noverlap=2048,
        return_onesided=False,
        scaling='density'
    )

    Pxx = np.fft.fftshift(Pxx)
    center = len(Pxx) // 2
    Pxx[center] = Pxx[center-1]

    f = np.fft.fftshift(f)

    Pxx_db = 10 * np.log10(Pxx / np.max(Pxx) + 1e-15)

    plt.figure()
    plt.plot(f, Pxx_db)
    plt.xlabel("Frequency, Hz")
    plt.ylabel("PSD, dB")
    plt.title("Smoothed spectrum (Welch)")
    plt.grid(True)
    plt.show()

def main():
    # Clear state (Python equivalent)
    plt.close('all')
    
    # Parameters
    prm = {
        'sizeSig': int(1e5),
        'txFs': 100e6,      # 100 MHz
        'sigBand': 20e6,    # 20 MHz bandwidth
        'up': 4             # Upsampling factor
    }
    
    # Signal generation
    sig = generator(prm)
    sigNorm = sig / np.max(np.abs(sig))
    
    # PA modeling
    sigAmp = amp_model(prm, sigNorm)
    sigAmpNorm = sigAmp / np.max(np.abs(sigAmp)) * np.max(np.abs(sig))
    ampGain = np.sqrt(np.mean(np.abs(sigAmp)**2) / np.mean(np.abs(sigNorm)**2))
    
    # CNN-DPD configuration
    prm['cnn'] = {
        'memory': 3,      # DPD memory depth
        'epochs': 20,
        'lr': 0.4,        # High learning rate as in original
        'ampGain': ampGain
    }
    
    # Train DPD
    sig_dpd, model = cnn_dpd(sig, sigAmp, prm)
    
    # Linearized output (DPD → PA)
    sigAmp_linear = amp_model(prm, sig_dpd) / ampGain
    
    f, PxxAmp = signal.welch(
        sigAmp/ampGain,
        fs=prm['txFs']*prm['up'],
        window='hann',
        nperseg=4096,
        noverlap=2048,
        return_onesided=False,
        scaling='density'
    )

    PxxAmp = np.fft.fftshift(PxxAmp)
    center = len(PxxAmp) // 2
    PxxAmp[center] = PxxAmp[center-1]
    
    fLinear, PxxLinear = signal.welch(
        sigAmp_linear,
        fs=prm['txFs']*prm['up'],
        window='hann',
        nperseg=4096,
        noverlap=2048,
        return_onesided=False,
        scaling='density'
    )

    PxxLinear = np.fft.fftshift(PxxLinear)
    center = len(PxxLinear) // 2
    PxxLinear[center] = PxxLinear[center-1]

    f = np.fft.fftshift(f)

    PxxAmp_db = 10 * np.log10(PxxAmp / np.max(PxxAmp) + 1e-15)
    PxxLinear_db = 10 * np.log10(PxxLinear / np.max(PxxLinear) + 1e-15)

    plt.figure()
    plt.plot(f / 1e6, PxxAmp_db, 'r', linewidth=1.5, label='Before DPD')
    plt.plot(f / 1e6, PxxLinear_db, 'b', linewidth=1.5, label='After DPD')
    plt.xlabel('Frequency, MHz')
    plt.ylabel('Magnitude, dB')
    plt.title('Power Spectral Density of generated signal')
    plt.xlim([-100, 100])
    plt.ylim([-100, 0])
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()