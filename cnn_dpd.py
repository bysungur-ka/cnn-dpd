import numpy as np
import matplotlib.pyplot as plt

def init_mse_monitor(target_dB):
    """Initialize training monitor with robust non-blocking mode"""
    plt.ion()  # Включаем интерактивный режим ДО создания фигуры
    fig = plt.figure('CNN-DPD Training')
    ax = plt.gca()
    ax.grid(True, alpha=0.3)
    h, = ax.plot([], [], 'm', linewidth=2)
    ax.axhline(y=target_dB, color='r', linestyle='--', linewidth=1.5, 
               label=f'Target {target_dB:.0f} dB')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (dB)')
    ax.set_ylim([-70, 0])
    ax.set_title('Training MSE')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)  # Критически важно для инициализации бэкенда
    return {'ax': ax, 'h': h, 'fig': fig, 'alive': True}


def update_mse_monitor(mon, ep, mse_hist):
    """Safe non-blocking update with window-close detection"""
    if not mon['alive']:
        return False
    
    try:
        # Проверяем, закрыто ли окно
        if not plt.fignum_exists(mon['fig'].number):
            mon['alive'] = False
            plt.ioff()  # Выключаем интерактивный режим
            print("\nMonitor window closed. Continuing training without visualization...")
            return False
        
        # Обновляем данные
        mon['h'].set_data(np.arange(1, ep + 1), mse_hist[:ep])
        mon['ax'].set_xlim([1, max(10, ep)])
        
        # Неблокирующее обновление
        mon['fig'].canvas.draw()
        mon['fig'].canvas.flush_events()
        plt.pause(0.001)  # Минимальная пауза для обработки событий GUI
        
        return True
        
    except Exception as e:
        mon['alive'] = False
        plt.ioff()
        print(f"\nMonitor error: {e}. Continuing without visualization...")
        return False


def cnn_dpd(sig_clean, sig_distorted, prm):
    """
    Pure ILA CNN-DPD
    CNN learns inverse PA:  y -> x
    """

    cnn_prm = prm.get('cnn', {})
    M = cnn_prm.get('memory', 3)
    M1 = cnn_prm.get('M1', 12)
    M2 = cnn_prm.get('M2', 8)
    lr = cnn_prm.get('lr', 0.01)
    epochs = cnn_prm.get('epochs', 200)

    K = M
    F = 8

    # -------------------------------------------------
    # Normalization
    # -------------------------------------------------
    sig_clean_n = sig_clean / np.max(np.abs(sig_clean))
    sig_distorted_n = sig_distorted / np.max(np.abs(sig_clean))

    N = len(sig_clean_n)

    # -------------------------------------------------
    # Initialize weights
    # -------------------------------------------------
    rng = np.random.RandomState(42)

    Wc = 0.05 * rng.randn(F, K, 2)
    bc = np.zeros((F, 1))

    W1 = 0.05 * rng.randn(M1, F)
    b1 = np.zeros((M1, 1))

    W2 = 0.05 * rng.randn(M2, M1)
    b2 = np.zeros((M2, 1))

    Wout = 0.05 * rng.randn(2, M2)
    bout = np.zeros((2, 1))

    mse_hist = np.zeros(epochs)

    print("Training CNN ILA (postdistorter)...")

    # ==========================================================
    # TRAINING LOOP (PURE ILA)
    # ==========================================================
    for ep in range(epochs):

        # -------------------------------------------------
        # CNN INPUT = distorted signal (PA output)
        # -------------------------------------------------
        Zc = np.zeros((F, N))

        for n in range(K - 1, N):
            for f in range(F):
                acc = 0.0
                for k in range(K):
                    idx = n - k
                    acc += (Wc[f, k, 0] * sig_distorted_n[idx].real +
                            Wc[f, k, 1] * sig_distorted_n[idx].imag)
                Zc[f, n] = acc + bc[f, 0]

        Ac = np.tanh(Zc)

        # -------------------------------------------------
        # Target = clean signal
        # -------------------------------------------------
        Y = np.vstack([sig_clean_n.real, sig_clean_n.imag])

        # -------------------------------------------------
        # FC forward
        # -------------------------------------------------
        A1 = np.tanh(W1 @ Ac + b1)
        A2 = np.tanh(W2 @ A1 + b2)
        Yh = Wout @ A2 + bout

        err = Yh - Y
        mse = np.mean(err**2)
        mse_hist[ep] = 10 * np.log10(mse + 1e-15)

        # -------------------------------------------------
        # Backprop
        # -------------------------------------------------
        dY = 2 * err / N

        dWout = dY @ A2.T
        dbout = np.sum(dY, axis=1, keepdims=True)

        dA2 = Wout.T @ dY
        dZ2 = dA2 * (1 - A2**2)
        dW2 = dZ2 @ A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (1 - A1**2)
        dW1 = dZ1 @ Ac.T
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        dAc = W1.T @ dZ1
        dZc = dAc * (1 - Ac**2)

        dbc = np.sum(dZc[:, K - 1:], axis=1, keepdims=True)

        dWc = np.zeros_like(Wc)
        for f in range(F):
            for k in range(K):
                n_idx = np.arange(K - 1, N)
                input_idx = n_idx - k

                dz_slice = dZc[f, n_idx]
                inp_real = sig_distorted_n[input_idx].real
                inp_imag = sig_distorted_n[input_idx].imag

                dWc[f, k, 0] = np.sum(dz_slice * inp_real) / N
                dWc[f, k, 1] = np.sum(dz_slice * inp_imag) / N

        dWc /= N
        dbc /= N

        # -------------------------------------------------
        # Update
        # -------------------------------------------------
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        Wout -= lr * dWout
        bout -= lr * dbout
        Wc -= lr * dWc
        bc -= lr * dbc

        print(f"Epoch {ep+1:4d} | ILA Loss = {mse_hist[ep]:.2f} dB")

    # ==========================================================
    # AFTER TRAINING → APPLY AS PREDISTORTER
    # ==========================================================
    print("Generating predistorted signal...")

    Zc_dpd = np.zeros((F, N))
    for n in range(K - 1, N):
        for f in range(F):
            acc = 0.0
            for k in range(K):
                idx = n - k
                acc += (Wc[f, k, 0] * sig_clean_n[idx].real +
                        Wc[f, k, 1] * sig_clean_n[idx].imag)
            Zc_dpd[f, n] = acc + bc[f, 0]

    Ac_dpd = np.tanh(Zc_dpd)
    A1_dpd = np.tanh(W1 @ Ac_dpd + b1)
    A2_dpd = np.tanh(W2 @ A1_dpd + b2)
    Y_dpd = Wout @ A2_dpd + bout

    sig_predist = Y_dpd[0, :] + 1j * Y_dpd[1, :]

    model = {
        'Wc': Wc, 'bc': bc,
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'Wout': Wout, 'bout': bout,
        'memory': M,
        'kernel': K,
        'filters': F
    }

    return sig_predist, model
