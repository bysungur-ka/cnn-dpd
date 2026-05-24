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
    mu: float = 1e-3,
    epochs: int = 20,
    w0: np.ndarray | None = None,
    print_every: int = 1,
    verbose: bool = True,
):
    """
    Обучение постдистортера на основе нормализованного LMS.

    Используется в ILA-схеме:
        вход модели  : y[n]  — выход усилителя;
        целевой сигнал: x[n] — вход усилителя / эталон после выравнивания.

    Модель:
        x_hat[n] = phi[n]^T a

    NLMS-обновление:
        a <- a + mu / (eps + ||phi[n]||^2) * conj(phi[n]) * e[n]

    Параметры:
        y            : комплексный вход постдистортера;
        x            : целевой комплексный сигнал;
        orders       : порядки нелинейности, например (1, 3, 5);
        memory_depth : глубина памяти;
        mu           : шаг адаптации NLMS;
        epochs       : число эпох;
        w0           : начальные коэффициенты;
        print_every  : как часто печатать лог;
        verbose      : печатать ли лог.

    Возвращает:
        a : комплексные коэффициенты модели.
    """
    import numpy as np

    eps = 1e-12

    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    n0 = min(len(y), len(x))
    y = y[:n0]
    x = x[:n0]

    if memory_depth < 1:
        raise ValueError("memory_depth must be >= 1")

    if len(y) <= memory_depth:
        raise ValueError("Signal is too short for selected memory_depth")

    # Матрица признаков MP:
    # phi[n] = z[n-m] |z[n-m]|^(k-1)
    Phi = build_mp_matrix(
        y,
        orders=orders,
        memory_depth=memory_depth,
    )

    # Первые memory_depth - 1 отсчетов не имеют полной истории
    start = memory_depth - 1
    Phi = Phi[start:, :]
    x = x[start:]

    n_samples, n_coeffs = Phi.shape

    # Нормировка столбцов матрицы признаков для численной устойчивости
    col_rms = np.sqrt(np.mean(np.abs(Phi) ** 2, axis=0) + eps)
    Phi_n = Phi / col_rms

    # Начальная инициализация
    if w0 is None:
        a_n = np.zeros(n_coeffs, dtype=np.complex128)

        # Инициализация линейного члена как тождественного преобразования.
        # Это полезно для ILA: начальная модель хотя бы пропускает сигнал.
        a_n[0] = 1.0 + 0.0j
    else:
        a_n = np.asarray(w0, dtype=np.complex128).reshape(-1)

        if a_n.size != n_coeffs:
            raise ValueError(f"w0 has length {a_n.size}, but expected {n_coeffs}")

        # Если w0 был задан в ненормированной шкале, переводим его
        # в шкалу нормированных признаков.
        a_n = a_n * col_rms

    if verbose:
        print("[LMS] Training postdistorter")
        print(f"[LMS] orders = {orders}")
        print(f"[LMS] memory_depth = {memory_depth}")
        print(f"[LMS] n_samples = {n_samples}")
        print(f"[LMS] n_coeffs = {n_coeffs}")
        print(f"[LMS] mu = {mu}")
        print(f"[LMS] epochs = {epochs}")

    for epoch in range(epochs):
        mse_acc = 0.0
        x_power_acc = 0.0

        for n in range(n_samples):
            phi = Phi_n[n, :]

            x_hat = phi @ a_n
            e = x[n] - x_hat

            phi_power = np.vdot(phi, phi).real + eps

            # NLMS update
            a_n = a_n + (mu / phi_power) * np.conj(phi) * e

            mse_acc += np.abs(e) ** 2
            x_power_acc += np.abs(x[n]) ** 2

            if not np.all(np.isfinite(a_n)):
                raise FloatingPointError(
                    f"LMS/NLMS diverged at epoch={epoch + 1}, sample={n}. "
                    f"Try smaller mu."
                )

        mse_epoch = mse_acc / n_samples
        nmse_epoch = 10.0 * np.log10((mse_acc + eps) / (x_power_acc + eps))
        coef_norm = np.linalg.norm(a_n)

        if verbose and (
            epoch == 0 or (epoch + 1) % print_every == 0 or (epoch + 1) == epochs
        ):
            print(
                f"[LMS] epoch {epoch + 1:4d}/{epochs}, "
                f"MSE = {mse_epoch:.3e}, "
                f"NMSE = {nmse_epoch:.2f} dB, "
                f"coef_norm = {coef_norm:.3e}"
            )

    # Возврат коэффициентов из шкалы нормированных признаков
    # к исходной матрице признаков
    a = a_n / col_rms

    return a
