import numpy as np

def amp_model(prm, x):
    """
    GMP-lite PA model:
      - Memory polynomial main branch
      - One envelope cross-term (lagging envelope) to emulate memory effects

    prm:
      pa_alpha   : nonlinearity level (default 0.3..1.0)
      pa_memory  : main memory depth M (default 3)
      gmp_k      : envelope lag k (default 2)
      gmp_beta   : cross-term strength (default 0.15)
      mem_decay  : optional exponential decay for memory taps (default 0.7)
    """

    c1 = 14.974 + 1j * 0.0519
    c3 = -23.0954 + 1j * 4.968
    c5 = 21.3936 + 1j * 0.4305

    alpha = float(prm.get('pa_alpha', 0.3))
    M = int(prm.get('pa_memory', 3))

    # GMP-lite cross-term controls
    k = int(prm.get('gmp_k', 2))          # envelope lag
    beta = float(prm.get('gmp_beta', 0.15))
    mem_decay = float(prm.get('mem_decay', 0.7))

    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    y = np.zeros_like(x)

    # Precompute envelope
    env2 = np.abs(x) ** 2

    # --- Main branch: memory polynomial over xm ---
    for m in range(M):
        w = mem_decay ** m

        xm = np.zeros_like(x)
        xm[m:] = x[:N - m]

        y += w * xm * (
            c1
            + alpha * c3 * np.abs(xm) ** 2
            + (alpha ** 2) * c5 * np.abs(xm) ** 4
        )

        # --- GMP-lite: lagging envelope cross-term ---
        # term ~ xm * |x[n-m-k]|^2  (uses envelope from an earlier sample)
        if k > 0:
            env_lag = np.zeros_like(env2)
            shift = m + k
            env_lag[shift:] = env2[:N - shift]
            y += w * (beta * alpha) * xm * env_lag

    return y