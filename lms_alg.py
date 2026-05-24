import numpy as np


def estimate_complex_gain(x_ref: np.ndarray, y: np.ndarray):
    """
    LS-оценка комплексного усиления G из модели:
        y ≈ G * x_ref
    """
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    n = min(len(x_ref), len(y))
    x_ref = x_ref[:n]
    y = y[:n]

    return np.vdot(x_ref, y) / (np.vdot(x_ref, x_ref) + 1e-15)


def build_mp_matrix(z: np.ndarray, orders=(1, 3, 5), memory_depth: int = 3):
    """
    Memory Polynomial basis:
        phi_{p,m}[n] = z[n-m] * |z[n-m]|^(p-1)

    Возвращает матрицу размера (N, P*M), где:
      P = len(orders)
      M = memory_depth
    """
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
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


def apply_mp_predistorter(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Применение MP-предысказителя:
        u[n] = sum_{m,p} a_{p,m} x[n-m] |x[n-m]|^(p-1)
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    a = np.asarray(a, dtype=np.complex128).reshape(-1)

    phi = build_mp_matrix(x, orders=orders, memory_depth=memory_depth)

    if phi.shape[1] != len(a):
        raise ValueError(
            f"Coefficient length mismatch: Phi has {phi.shape[1]} columns, "
            f"but len(a) = {len(a)}. Check orders and memory_depth."
        )

    return phi @ a


def nmse_db(y_hat: np.ndarray, y_ref: np.ndarray):
    """
    NMSE без gain alignment.
    """
    y_hat = np.asarray(y_hat, dtype=np.complex128).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=np.complex128).reshape(-1)

    n = min(len(y_hat), len(y_ref))
    y_hat = y_hat[:n]
    y_ref = y_ref[:n]

    e = y_hat - y_ref
    num = np.mean(np.abs(e) ** 2) + 1e-15
    den = np.mean(np.abs(y_ref) ** 2) + 1e-15

    return 10.0 * np.log10(num / den)


def nmse_db_gain_aligned(y_hat: np.ndarray, y_ref: np.ndarray):
    """
    NMSE с комплексным gain alignment:
        y_hat_aligned = alpha * y_hat
    """
    y_hat = np.asarray(y_hat, dtype=np.complex128).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=np.complex128).reshape(-1)

    n = min(len(y_hat), len(y_ref))
    y_hat = y_hat[:n]
    y_ref = y_ref[:n]

    alpha = np.vdot(y_hat, y_ref) / (np.vdot(y_hat, y_hat) + 1e-15)
    y_hat_aligned = alpha * y_hat

    return nmse_db(y_hat_aligned, y_ref)


def normalize_drive_rms(x_ref: np.ndarray, u: np.ndarray):
    """
    RMS-нормировка drive-сигнала u к уровню исходного сигнала x_ref.

    Возвращает:
        u_norm        : нормированный сигнал
        p_ref         : RMS power исходного сигнала
        p_before_norm : RMS power сигнала u до нормировки
    """
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    u = np.asarray(u, dtype=np.complex128).reshape(-1)

    n = min(len(x_ref), len(u))
    x_ref = x_ref[:n]
    u = u[:n]

    p_ref = np.mean(np.abs(x_ref) ** 2) + 1e-15
    p_u = np.mean(np.abs(u) ** 2) + 1e-15

    u_norm = u * np.sqrt(p_ref / p_u)

    return u_norm, p_ref, p_u


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
    verbose: bool = True,
    w0: np.ndarray | None = None,
    keep_best: bool = True,
    **kwargs,
):
    """
    LMS/NLMS-оценка коэффициентов postdistorter-а с MP-базисом:

        x_hat[n] = sum_{m,p} a_{p,m} y_eff[n-m] |y_eff[n-m]|^(p-1)

    где:
        y_eff = y / G, если normalize_gain=True.

    В ILA:
        y — выход усилителя;
        x — вход усилителя / целевой сигнал.

    Параметры w0, keep_best, verbose и **kwargs добавлены для совместимости
    с main.py, если они уже передаются из prm.
    """
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    n0 = min(len(y), len(x))
    y = y[:n0]
    x = x[:n0]

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
    if w0 is None:
        a = np.zeros(n_features, dtype=np.complex128)
        a[0] = 1.0 + 0.0j
    else:
        a = np.asarray(w0, dtype=np.complex128).reshape(-1)

        if len(a) != n_features:
            # Если размерность поменялась, лучше не падать, а начать заново.
            if verbose:
                print(f"[LMS] w0 ignored: len(w0)={len(a)}, " f"expected {n_features}")
            a = np.zeros(n_features, dtype=np.complex128)
            a[0] = 1.0 + 0.0j

    eps = 1e-12
    idx = np.arange(n_samples)

    best_a = a.copy()
    best_nmse = np.inf

    if verbose:
        print("[LMS] Training postdistorter")
        print(f"[LMS] orders = {orders}")
        print(f"[LMS] memory_depth = {memory_depth}")
        print(f"[LMS] n_samples = {n_samples}")
        print(f"[LMS] n_coeffs = {n_features}")
        print(f"[LMS] mu = {mu}")
        print(f"[LMS] epochs = {epochs}")
        print(f"[LMS] normalize_gain = {normalize_gain}")
        print(f"[LMS] normalized = {normalized}")

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

            if not np.all(np.isfinite(a)):
                raise FloatingPointError(
                    f"LMS/NLMS diverged at epoch={epoch}, sample={n}. "
                    f"Try smaller mu."
                )

        x_hat = phi @ a
        train_nmse = nmse_db(x_hat, x)

        if train_nmse < best_nmse:
            best_nmse = train_nmse
            best_a = a.copy()

        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            coef_norm = np.linalg.norm(a)
            print(
                f" Epoch {epoch:5d}/{epochs} | "
                f"Train NMSE(u)={train_nmse:.2f} dB | "
                f"best={best_nmse:.2f} dB | "
                f"coef_norm={coef_norm:.3e}"
            )

    if keep_best:
        a = best_a

    if return_gain:
        return a, G

    return a


def train_lms_ila_predistorter(
    x_al: np.ndarray,
    y_al: np.ndarray,
    pa_fn,
    lms_prm: dict | None = None,
):
    """
    Простая ILA-обертка для LMS/MP-предысказителя.

    Сделана так, чтобы можно было поменять только lms_alg.py.

    Логика:
      1. Первая итерация обучается на паре x_al -> y_al.
      2. На каждой итерации обучается postdistorter:
             y_i -> u_i
         При normalize_gain=True нормировка y_i/G выполняется внутри
         lms_postdistorter_coeffs().
      3. Полученные коэффициенты применяются как predistorter:
             x_al -> u_next
      4. u_next нормируется по RMS.
      5. u_next прогоняется через PA.
      6. Следующая итерация обучается на новой паре u_next -> y_next.
    """
    if lms_prm is None:
        lms_prm = {}

    orders = lms_prm.get("orders", (1, 3, 5))
    memory_depth = int(lms_prm.get("memory_depth", 3))
    mu = float(lms_prm.get("mu", 0.05))
    epochs = int(lms_prm.get("epochs", 30))
    ila_iters = int(lms_prm.get("ila_iters", 1))

    normalize_gain = bool(lms_prm.get("normalize_gain", True))
    normalized = bool(lms_prm.get("normalized", True))
    shuffle = bool(lms_prm.get("shuffle", False))
    print_every = int(lms_prm.get("print_every", 1))
    verbose = bool(lms_prm.get("verbose", True))
    keep_best = bool(lms_prm.get("keep_best", True))

    x_ref = np.asarray(x_al, dtype=np.complex128).reshape(-1)
    y_cur = np.asarray(y_al, dtype=np.complex128).reshape(-1)

    n0 = min(len(x_ref), len(y_cur))
    x_ref = x_ref[:n0]
    y_cur = y_cur[:n0]

    # Первая итерация: вход PA равен исходному сигналу.
    u_cur = x_ref.copy()

    a = None
    G_hist = []
    nmse_post_hist = []
    nmse_after_hist = []
    drive_power_hist = []

    for ila_idx in range(ila_iters):
        if verbose:
            print("")
            print(f"[LMS-ILA] iteration {ila_idx + 1}/{ila_iters}")

        n = min(len(u_cur), len(y_cur))
        u_train = u_cur[:n]
        y_train = y_cur[:n]

        # Обучаем postdistorter y_train -> u_train.
        # ВАЖНО: y/G делается внутри lms_postdistorter_coeffs(),
        # если normalize_gain=True.
        a, G = lms_postdistorter_coeffs(
            y_train,
            u_train,
            orders=orders,
            memory_depth=memory_depth,
            mu=mu,
            epochs=epochs,
            normalize_gain=normalize_gain,
            normalized=normalized,
            shuffle=shuffle,
            print_every=print_every,
            return_gain=True,
            verbose=verbose,
            w0=None,
            keep_best=keep_best,
        )

        G_hist.append(G)

        # Диагностика postdistorter на текущей паре.
        if normalize_gain:
            y_eff = y_train / (G + 1e-15)
        else:
            y_eff = y_train

        u_hat = apply_mp_predistorter(
            y_eff,
            a,
            orders=orders,
            memory_depth=memory_depth,
        )

        post_nmse = nmse_db(u_hat, u_train)
        nmse_post_hist.append(post_nmse)

        if verbose:
            print(f"[LMS-ILA] postdistorter NMSE = {post_nmse:.2f} dB")

        # Перенос коэффициентов в прямой тракт: x_ref -> u_next.
        u_next = apply_mp_predistorter(
            x_ref,
            a,
            orders=orders,
            memory_depth=memory_depth,
        )

        # Нормировка drive-сигнала.
        u_next, p_ref, p_before_norm = normalize_drive_rms(x_ref, u_next)
        drive_power_hist.append(10.0 * np.log10(p_before_norm + 1e-15))

        if verbose:
            print(
                f"[LMS-ILA] drive RMS before norm = "
                f"{10.0 * np.log10(p_before_norm + 1e-15):.2f} dB"
            )

        # Обратная связь через PA.
        y_next = pa_fn(u_next)

        # Диагностика системного результата после PA.
        n_eval = min(len(x_ref), len(y_next))
        y_eval = y_next[:n_eval]
        x_eval = x_ref[:n_eval]

        after_nmse = nmse_db_gain_aligned(y_eval, x_eval)
        nmse_after_hist.append(after_nmse)

        if verbose:
            print(f"[LMS-ILA] After-PA gain-aligned NMSE = " f"{after_nmse:.2f} dB")

        # Следующая итерация обучается на новой паре.
        u_cur = u_next
        y_cur = y_next

    # Финальное применение DPD.
    x_dpd = apply_mp_predistorter(
        x_ref,
        a,
        orders=orders,
        memory_depth=memory_depth,
    )

    x_dpd, p_ref, p_before_norm = normalize_drive_rms(x_ref, x_dpd)

    if verbose:
        print(
            f"[LMS-ILA] final drive RMS before norm = "
            f"{10.0 * np.log10(p_before_norm + 1e-15):.2f} dB"
        )

    model = {
        "a": a,
        "orders": orders,
        "memory_depth": memory_depth,
        "mu": mu,
        "epochs": epochs,
        "ila_iters": ila_iters,
        "normalize_gain": normalize_gain,
        "normalized": normalized,
        "shuffle": shuffle,
        "keep_best": keep_best,
        "G_hist": np.asarray(G_hist),
        "nmse_post_hist_db": np.asarray(nmse_post_hist),
        "nmse_after_hist_db": np.asarray(nmse_after_hist),
        "drive_power_before_norm_db": np.asarray(drive_power_hist),
        "kind": "lms_mp_ila",
    }

    return x_dpd, model
