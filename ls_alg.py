import numpy as np


def align_by_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 300):
    """
    Грубое выравнивание x и y по максимальной корреляции (по модулю).

    Возвращает (x_al, y_al, lag), где:
      lag > 0  => y задержан относительно x на lag
      lag < 0  => y опережает x
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    c = np.empty_like(lags, dtype=np.float64)

    for i, lag in enumerate(lags):
        if lag < 0:
            c[i] = np.abs(np.vdot(x[-lag:], y[:N + lag]))
        elif lag > 0:
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


def estimate_complex_gain(x_ref: np.ndarray, y: np.ndarray):
    """
    LS-оценка комплексного усиления G из модели:
        y ≈ G * x_ref
    """
    x_ref = np.asarray(x_ref, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    return np.vdot(x_ref, y) / (np.vdot(x_ref, x_ref) + 1e-15)


def build_poly_matrix(z: np.ndarray, orders=(1, 3, 5)):
    """
    Старый memoryless polynomial basis:
        z * |z|^(p-1)
    Оставлен для совместимости.
    """
    z = np.asarray(z, dtype=np.complex128)
    absz = np.abs(z)
    cols = [z * (absz ** (p - 1)) for p in orders]
    return np.vstack(cols).T  # (N, P)


def build_mp_matrix(z: np.ndarray, orders=(1, 3, 5), memory_depth: int = 3):
    """
    Memory Polynomial basis:
        phi_{p,m}[n] = z[n-m] * |z[n-m]|^(p-1)

    Возвращает матрицу размера (N, P*M), где:
      P = len(orders)
      M = memory_depth

    Используется каузальная zero-padding схема:
      для n < m соответствующий delayed sample считается нулём.
    """
    z = np.asarray(z, dtype=np.complex128)
    N = len(z)
    M = int(memory_depth)

    if M <= 0:
        raise ValueError("memory_depth must be >= 1")

    cols = []

    for m in range(M):
        zm = np.zeros_like(z)
        if m == 0:
            zm[:] = z
        else:
            zm[m:] = z[:N - m]

        abszm = np.abs(zm)
        for p in orders:
            cols.append(zm * (abszm ** (p - 1)))

    return np.vstack(cols).T  # (N, P*M)


def ls_postdistorter_coeffs(
    y: np.ndarray,
    x: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
    ridge: float = 0.0,
    normalize_gain: bool = True,
    return_gain: bool = False,
):
    """
    Solve complex LS postdistorter with Memory Polynomial basis:
        min ||Phi(y_eff) a - x||^2

    where
        y_eff = y / G,   if normalize_gain=True
        G = <x, y> / <x, x>

    Дополнительно:
      1) используется только валидная часть сигнала
         (без первых memory_depth-1 строк с zero-padding)
      2) выполняется RMS-нормировка столбцов Phi перед LS
         для улучшения обусловленности

    Parameters
    ----------
    y : ndarray
        PA output (aligned)
    x : ndarray
        PA input / reference target (aligned)
    orders : tuple
        Polynomial orders
    memory_depth : int
        Memory depth M of MP model
    ridge : float
        Ridge regularization
    normalize_gain : bool
        If True, first normalize y by complex gain G
    return_gain : bool
        If True, return (a, G) instead of only a
    """
    y = np.asarray(y, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128)

    G = 1.0 + 0.0j
    if normalize_gain:
        G = estimate_complex_gain(x, y)
        y_eff = y / (G + 1e-15)
    else:
        y_eff = y

    # Полная MP-матрица
    Phi = build_mp_matrix(y_eff, orders=orders, memory_depth=memory_depth)

    # -------------------------------------------------
    # Пункт 3: берём только валидную часть
    # -------------------------------------------------
    start = max(0, int(memory_depth) - 1)
    Phi_v = Phi[start:, :]
    x_v = x[start:]

    # -------------------------------------------------
    # Пункт 2: нормировка столбцов Phi
    # -------------------------------------------------
    col_rms = np.sqrt(np.mean(np.abs(Phi_v) ** 2, axis=0) + 1e-15)
    Phi_n = Phi_v / col_rms

    P = Phi_n.shape[1]
    A = Phi_n.conj().T @ Phi_n
    b = Phi_n.conj().T @ x_v

    if ridge > 0.0:
        A = A + ridge * np.eye(P, dtype=np.complex128)

    a_n = np.linalg.solve(A, b)

    # Возвращаем коэффициенты в исходный масштаб
    a = a_n / col_rms

    if return_gain:
        return a, G
    return a


def apply_predistorter(
    x_ref: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Применение MP predistorter:
        x_dpd[n] = sum_{m,p} a_{p,m} x_ref[n-m] |x_ref[n-m]|^(p-1)
    """
    Phi = build_mp_matrix(x_ref, orders=orders, memory_depth=memory_depth)
    return Phi @ a


def nmse_db(y_hat: np.ndarray, y_ref: np.ndarray):
    e = y_hat - y_ref
    num = np.mean(np.abs(e) ** 2) + 1e-15
    den = np.mean(np.abs(y_ref) ** 2) + 1e-15
    return 10.0 * np.log10(num / den)


def nmse_db_gain_aligned(y, x):
    """
    Gain-aligned NMSE:
        y ≈ alpha * x
    """
    alpha = np.vdot(x, y) / (np.vdot(x, x) + 1e-15)
    e = y - alpha * x
    return 10 * np.log10(
        (np.mean(np.abs(e) ** 2) + 1e-15) /
        (np.mean(np.abs(alpha * x) ** 2) + 1e-15)
    )