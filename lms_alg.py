import numpy as np


def estimate_complex_gain(x_ref: np.ndarray, y: np.ndarray):
    """
    LS-оценка комплексного усиления G из модели:
        y ≈ G * x_ref
    """
    x_ref = np.asarray(x_ref, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    return np.vdot(x_ref, y) / (np.vdot(x_ref, x_ref) + 1e-15)


def build_mp_matrix(z: np.ndarray, orders=(1, 3, 5), memory_depth: int = 3):
    """
    Memory Polynomial basis:
        phi_{p,m}[n] = z[n-m] * |z[n-m]|^(p-1)

    Возвращает матрицу размера (N, P*M), где:
      P = len(orders)
      M = memory_depth
    """
    z = np.asarray(z, dtype=np.complex128)
    n_samples = len(z)
    memory_depth = int(memory_depth)

    if memory_depth <= 0:
        raise ValueError("memory_depth must be >= 1")

    cols = []

    for m in range(memory_depth):
        zm = np.zeros_like(z)
        if m == 0:
            zm[:] = z
        else:
            zm[m:] = z[: n_samples - m]

        abs_zm = np.abs(zm)
        for p in orders:
            cols.append(zm * (abs_zm ** (p - 1)))

    return np.vstack(cols).T


def nmse_db(y_hat: np.ndarray, y_ref: np.ndarray):
    e = y_hat - y_ref
    num = np.mean(np.abs(e) ** 2) + 1e-15
    den = np.mean(np.abs(y_ref) ** 2) + 1e-15
    return 10.0 * np.log10(num / den)


def lms_postdistorter_coeffs(
    y: np.ndarray,
    x: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
    mu: float = 0.05,
    epochs: int = 30,
    normalize_gain: bool = True,
    normalized: bool = True,
    shuffle: bool = False,
    print_every: int = 1,
    return_gain: bool = False,
):
    """
    LMS/NLMS-оценка коэффициентов postdistorter-а с MP-базисом:

        x_hat[n] = sum_{m,p} a_{p,m} y_eff[n-m] |y_eff[n-m]|^(p-1)

    где
        y_eff = y / G, если normalize_gain=True

    Parameters
    ----------
    y : ndarray
        PA output (aligned)
    x : ndarray
        PA input / reference target (aligned)
    orders : tuple
        Polynomial orders
    memory_depth : int
        MP memory depth
    mu : float
        Шаг LMS. Для normalized=True обычно 0.01...0.1
    epochs : int
        Число проходов по данным
    normalize_gain : bool
        Нормировать ли y по комплексному gain
    normalized : bool
        Если True, используется NLMS
    shuffle : bool
        Перемешивать ли отсчёты внутри эпохи
    print_every : int
        Печать каждые print_every эпох
    return_gain : bool
        Если True, вернуть (a, G)
    """
    y = np.asarray(y, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128)

    if len(y) != len(x):
        raise ValueError("x and y must have the same length")

    G = 1.0 + 0.0j
    if normalize_gain:
        G = estimate_complex_gain(x, y)
        y_eff = y / (G + 1e-15)
    else:
        y_eff = y

    phi = build_mp_matrix(y_eff, orders=orders, memory_depth=memory_depth)
    n_samples, n_features = phi.shape

    # Инициализация: линейный коэффициент = 1, остальные = 0
    a = np.zeros(n_features, dtype=np.complex128)
    a[0] = 1.0 + 0.0j

    eps = 1e-12
    idx = np.arange(n_samples)

    for epoch in range(1, epochs + 1):
        if shuffle:
            np.random.shuffle(idx)

        for n in idx:
            phi_n = phi[n]
            x_hat_n = phi_n @ a
            e_n = x[n] - x_hat_n

            if normalized:
                step = mu / (np.vdot(phi_n, phi_n).real + eps)
            else:
                step = mu

            a = a + step * np.conj(phi_n) * e_n

        if (epoch == 1) or (epoch % print_every == 0) or (epoch == epochs):
            x_hat = phi @ a
            train_nmse = nmse_db(x_hat, x)
            print(f" Epoch {epoch:5d}/{epochs} | Train NMSE(u)={train_nmse:.2f} dB")

    if return_gain:
        return a, G
    return a