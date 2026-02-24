import numpy as np


# ----------------------------
# Initializers / activations
# ----------------------------
def _he_normal(rng, fan_in, shape):
    """Kaiming/He normal init for ReLU."""
    return rng.randn(*shape) * np.sqrt(2.0 / fan_in)


def _relu(z):
    return np.maximum(0.0, z)


def _drelu(z):
    return (z > 0.0).astype(z.dtype)


def _make_feature_channels(u: np.ndarray, mode: str):
    """
    Convert complex u into real-valued feature channels Xc (C,N):
    mode="iq":   [Re(u), Im(u)]  -> C=2
    mode="poly": [Re(u),Im(u), Re(u|u|^2),Im(u|u|^2), Re(u|u|^4),Im(u|u|^4)] -> C=6
    """
    u = np.asarray(u, dtype=np.complex128)
    ur = u.real
    ui = u.imag

    if mode.lower() == "iq":
        return np.vstack([ur, ui])  # (2,N)

    if mode.lower() == "poly":
        a2 = np.abs(u) ** 2
        u2 = u * a2           # u|u|^2
        u4 = u * (a2 ** 2)    # u|u|^4
        return np.vstack([ur, ui, u2.real, u2.imag, u4.real, u4.imag])  # (6,N)

    raise ValueError('features must be "iq" or "poly"')


def cnn_dpd(sig_clean: np.ndarray, sig_distorted: np.ndarray, prm: dict):
    """
    ILA CNN-DPD (numpy), ReLU + per-channel RMS normalization:
      - Train postdistorter g: y -> x
      - Apply as predistorter: x_dpd = g(x)

    Architecture (proper multi-channel 1D CNN):
      Conv1D (C channels -> F filters) + ReLU
      FC + ReLU
      FC + ReLU
      Linear(2)
    """

    cnn_prm = prm.get('cnn', {})
    M = int(cnn_prm.get('memory', 1))
    K = M
    F = int(cnn_prm.get('filters', 32))
    M1 = int(cnn_prm.get('M1', 64))
    M2 = int(cnn_prm.get('M2', 64))
    lr = float(cnn_prm.get('lr', 1e-3))
    epochs = int(cnn_prm.get('epochs', 300))
    seed = int(cnn_prm.get('seed', 42))
    print_every = int(cnn_prm.get('print_every', 10))

    features = str(cnn_prm.get('features', 'poly'))  # "iq" or "poly"
    debug_stats = bool(cnn_prm.get('debug_stats', False))

    # optional stabilization knobs
    weight_decay = float(cnn_prm.get('weight_decay', 0.0))  # e.g. 1e-6
    clip = float(cnn_prm.get('clip', 0.0))                  # e.g. 5.0 (0 disables)

    sig_clean = np.asarray(sig_clean, dtype=np.complex128)
    sig_distorted = np.asarray(sig_distorted, dtype=np.complex128)
    N = min(len(sig_clean), len(sig_distorted))
    sig_clean = sig_clean[:N]
    sig_distorted = sig_distorted[:N]

    # consistent scale for complex signals
    scale = np.max(np.abs(sig_clean)) + 1e-15
    x = sig_clean / scale
    y = sig_distorted / scale

    # targets in I/Q
    Y = np.vstack([x.real, x.imag])   # (2,N)

    # valid region
    n_valid = np.arange(K - 1, N, dtype=int)
    N_valid = len(n_valid)
    if N_valid <= 0:
        raise ValueError("Not enough samples for given memory/kernel length.")

    # feature channels for y
    Xy = _make_feature_channels(y, features)  # (C,N)
    C = Xy.shape[0]

    # ----------------------------
    # Per-channel RMS normalization (CRITICAL)
    # ----------------------------
    eps = 1e-12
    feat_rms = np.sqrt(np.mean(Xy[:, n_valid] ** 2, axis=1, keepdims=True) + eps)  # (C,1)
    Xy = Xy / feat_rms

    rng = np.random.RandomState(seed)

    # He init for ReLU
    Wc = _he_normal(rng, fan_in=C * K, shape=(F, K, C))
    bc = np.zeros((F, 1))

    W1 = _he_normal(rng, fan_in=F, shape=(M1, F))
    b1 = np.zeros((M1, 1))

    W2 = _he_normal(rng, fan_in=M1, shape=(M2, M1))
    b2 = np.zeros((M2, 1))

    Wout = _he_normal(rng, fan_in=M2, shape=(2, M2))
    bout = np.zeros((2, 1))

    print(f"Training CNN ILA (ReLU) y->x, features={features}, C={C}, K={K}...")

    def _clip_inplace(G):
        nrm = np.linalg.norm(G)
        if nrm > clip:
            G *= (clip / (nrm + 1e-15))

    for ep in range(epochs):
        # ---------- forward conv ----------
        Zc = np.zeros((F, N), dtype=np.float64)
        for k in range(K):
            idx = n_valid - k
            Zc[:, n_valid] += (Wc[:, k, :] @ Xy[:, idx])
        Zc[:, n_valid] += bc
        Ac = _relu(Zc)

        # ---------- forward FC ----------
        Z1 = W1 @ Ac + b1
        A1 = _relu(Z1)

        Z2 = W2 @ A1 + b2
        A2 = _relu(Z2)

        Yh = Wout @ A2 + bout  # (2,N)

        if debug_stats and (ep == 0 or (ep + 1) % print_every == 0):
            print("max|Zc| =", float(np.max(np.abs(Zc[:, n_valid]))))
            print("max|A1| =", float(np.max(np.abs(A1[:, n_valid]))))

        # ---------- loss on valid ----------
        err = Yh[:, n_valid] - Y[:, n_valid]
        mse = np.mean(err ** 2)

        p_ref = np.mean(Y[0, n_valid] ** 2 + Y[1, n_valid] ** 2) + 1e-15
        nmse = mse / p_ref

        if (ep + 1) % print_every == 0 or ep == 0:
            print(f"Epoch {ep+1:4d} | MSE={10*np.log10(mse+1e-15):.2f} dB | NMSE={10*np.log10(nmse+1e-15):.2f} dB")

        # ---------- backprop ----------
        dYh = np.zeros_like(Yh)
        dY = err / N_valid
        dYh[:, n_valid] = dY

        dWout = dYh @ A2.T
        dbout = np.sum(dYh, axis=1, keepdims=True)

        dA2 = Wout.T @ dYh
        dZ2 = dA2 * _drelu(Z2)
        dW2 = dZ2 @ A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * _drelu(Z1)
        dW1 = dZ1 @ Ac.T
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        dAc = W1.T @ dZ1
        dZc = dAc * _drelu(Zc)

        dbc = np.sum(dZc[:, n_valid], axis=1, keepdims=True) / N_valid

        dWc = np.zeros_like(Wc)
        dz = dZc[:, n_valid]  # (F,N_valid)
        for k in range(K):
            idx = n_valid - k
            Xslice = Xy[:, idx]            # (C, N_valid)
            dWc[:, k, :] = (dz @ Xslice.T) / N_valid

        # ---------- weight decay ----------
        if weight_decay > 0.0:
            dWout += weight_decay * Wout
            dW2 += weight_decay * W2
            dW1 += weight_decay * W1
            dWc += weight_decay * Wc

        # ---------- gradient clip ----------
        if clip > 0.0:
            _clip_inplace(dWout); _clip_inplace(dW2); _clip_inplace(dW1); _clip_inplace(dWc)
            _clip_inplace(dbout); _clip_inplace(db2); _clip_inplace(db1); _clip_inplace(dbc)

        # ---------- update ----------
        Wout -= lr * dWout
        bout -= lr * dbout
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        Wc -= lr * dWc
        bc -= lr * dbc

    print("Generating predistorted signal (apply g to clean input)...")

    # ---------- apply g to x (predistorter) ----------
    Xx = _make_feature_channels(x, features)     # (C,N)
    Xx = Xx / feat_rms                           # IMPORTANT: same normalization as training

    Zc_dpd = np.zeros((F, N), dtype=np.float64)
    for k in range(K):
        idx = n_valid - k
        Zc_dpd[:, n_valid] += (Wc[:, k, :] @ Xx[:, idx])
    Zc_dpd[:, n_valid] += bc
    Ac_dpd = _relu(Zc_dpd)

    Z1_dpd = W1 @ Ac_dpd + b1
    A1_dpd = _relu(Z1_dpd)
    
    print(np.max(np.abs(Zc_dpd))) 
    print(np.max(np.abs(A1_dpd)))

    Z2_dpd = W2 @ A1_dpd + b2
    A2_dpd = _relu(Z2_dpd)

    Y_dpd = Wout @ A2_dpd + bout

    x_dpd_n = Y_dpd[0, :] + 1j * Y_dpd[1, :]
    sig_predist = x_dpd_n * scale

    model = {
        'Wc': Wc, 'bc': bc,
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'Wout': Wout, 'bout': bout,
        'memory': M, 'kernel': K, 'filters': F,
        'features': features,
        'feat_rms': feat_rms,
        'scale': scale
    }

    return sig_predist, model