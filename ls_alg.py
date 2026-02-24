import numpy as np

def align_by_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 300):
    """
    Грубое выравнивание x и y по максимальной корреляции (по модулю).
    Возвращает (x_al, y_al, lag), где lag означает:
      lag > 0  => y задержан относительно x на lag (т.е. y[n+lag] ~ x[n])
      lag < 0  => y опережает x
    """
    x = np.asarray(x).astype(np.complex128)
    y = np.asarray(y).astype(np.complex128)

    # Ограничиваем длину для устойчивости (если очень длинные)
    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    c = np.empty_like(lags, dtype=np.float64)

    # vdot делает conj(first) * second
    for i, lag in enumerate(lags):
        if lag < 0:
            # y раньше: сравниваем x[-lag:] с y[:N+lag]
            c[i] = np.abs(np.vdot(x[-lag:], y[:N + lag]))
        elif lag > 0:
            # y позже: сравниваем x[:N-lag] с y[lag:]
            c[i] = np.abs(np.vdot(x[:N - lag], y[lag:]))
        else:
            c[i] = np.abs(np.vdot(x, y))

    lag0 = int(lags[int(np.argmax(c))])

    if lag0 < 0:
        x_al = x[-lag0:]
        y_al = y[:len(x_al)]
    elif lag0 > 0:
        x_al = x[:N - lag0]
        y_al = y[lag0:lag0 + len(x_al)]
    else:
        x_al = x
        y_al = y

    return x_al, y_al, lag0


def build_poly_matrix(z: np.ndarray, orders=(1, 3, 5)):
    z = np.asarray(z).astype(np.complex128)
    absz = np.abs(z)
    cols = [z * (absz ** (p - 1)) for p in orders]
    return np.vstack(cols).T  # (N,P)

def ls_postdistorter_coeffs(y: np.ndarray, x: np.ndarray, orders=(1, 3, 5), ridge: float = 0.0):
    """
    Solve: min ||Phi a - x||^2  (complex LS)
    Normal eq: (Phi^H Phi + ridge I) a = Phi^H x
    """
    Phi = build_poly_matrix(y, orders=orders)  # (N,P)
    x = np.asarray(x).astype(np.complex128)

    P = Phi.shape[1]
    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ x

    if ridge > 0.0:
        A = A + ridge * np.eye(P)

    a = np.linalg.solve(A, b)  
    return a

def apply_predistorter(x_ref: np.ndarray, a: np.ndarray, orders=(1, 3, 5)):
    Phi = build_poly_matrix(x_ref, orders=orders)
    return Phi @ a

def nmse_db(y_hat: np.ndarray, y_ref: np.ndarray):
    e = y_hat - y_ref
    num = np.mean(np.abs(e) ** 2) + 1e-15
    den = np.mean(np.abs(y_ref) ** 2) + 1e-15
    return 10.0 * np.log10(num / den)

def nmse_db_gain_aligned(y, x):
    # y ≈ alpha x
    alpha = np.vdot(x, y) / (np.vdot(x, x) + 1e-15)  # conj(x)^T y / conj(x)^T x
    e = y - alpha * x
    return 10*np.log10((np.mean(np.abs(e)**2)+1e-15) / (np.mean(np.abs(alpha*x)**2)+1e-15))