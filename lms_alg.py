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
    Gain-aligned NMSE.

    Сначала y_hat приводится к y_ref по комплексному коэффициенту,
    потом считается NMSE.
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
        u_norm : нормированный сигнал
        p_ref  : мощность x_ref
        p_u    : мощность u до нормировки
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


def apply_mp_predistorter(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Обычное применение MP-предысказителя:
        u[n] = Phi_x[n] @ a
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    a = np.asarray(a, dtype=np.complex128).reshape(-1)

    phi = build_mp_matrix(
        x,
        orders=orders,
        memory_depth=memory_depth,
    )

    if phi.shape[1] != len(a):
        raise ValueError(
            f"Coefficient length mismatch: Phi has {phi.shape[1]} columns, "
            f"but len(a) = {len(a)}. Check orders and memory_depth."
        )

    return phi @ a


def apply_mp_correction(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Вычисляет MP-поправку:
        c[n] = Phi_x[n] @ a
    """
    return apply_mp_predistorter(
        x,
        a,
        orders=orders,
        memory_depth=memory_depth,
    )


def apply_mp_feedback_predistorter(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Residual / feedback MP-предысказитель:
        u[n] = x[n] + c[n],
        c[n] = Phi_x[n] @ a
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    correction = apply_mp_correction(
        x,
        a,
        orders=orders,
        memory_depth=memory_depth,
    )

    return x + correction


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
    keep_best: bool = True,
):
    """
    LMS/NLMS-оценка коэффициентов postdistorter-а с MP-базисом.

    Модель:
        x_hat[n] = sum_{m,p} a_{p,m} y_eff[n-m] |y_eff[n-m]|^(p-1)

    где:
        y_eff = y / G, если normalize_gain=True.

    Эта функция оставлена как стабильная однопроходная ILA-реализация:
        y -> x.
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

    phi = build_mp_matrix(
        y_eff,
        orders=orders,
        memory_depth=memory_depth,
    )

    n_samples, n_features = phi.shape

    # Инициализация: линейный коэффициент = 1, остальные = 0
    a = np.zeros(n_features, dtype=np.complex128)
    a[0] = 1.0 + 0.0j

    best_a = a.copy()
    best_nmse = np.inf

    eps = 1e-12
    idx = np.arange(n_samples)

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

        if (epoch == 1) or (epoch % print_every == 0) or (epoch == epochs):
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


def train_lms_feedback_predistorter(
    x_al: np.ndarray,
    y_al: np.ndarray,
    pa_fn,
    lms_prm: dict | None = None,
):
    """
    LMS-предысказитель с обратной связью через PA.

    В отличие от ILA-постдистортера y -> u, здесь обучается
    residual-поправка к исходному сигналу:

        u_i[n] = x[n] + c_i[n]

    После каждого прогона через PA вычисляется ошибка:

        e_i[n] = x[n] - y_i[n] / G_i

    и формируется новая целевая поправка:

        c_target[n] = c_i[n] + beta * e_i[n]

    Далее LMS/NLMS обучает MP-модель:
        Phi_x[n] @ a ≈ c_target[n]
    """
    if lms_prm is None:
        lms_prm = {}

    orders = tuple(lms_prm.get("orders", (1, 3, 5)))
    memory_depth = int(lms_prm.get("memory_depth", 3))
    mu = float(lms_prm.get("mu", 0.01))
    epochs = int(lms_prm.get("epochs", 10))

    # Используем ila_iters как число feedback-итераций,
    # чтобы не менять имя параметра в prm.
    fb_iters = int(lms_prm.get("ila_iters", lms_prm.get("fb_iters", 5)))

    beta = float(lms_prm.get("feedback_beta", 0.1))
    normalized = bool(lms_prm.get("normalized", True))
    shuffle = bool(lms_prm.get("shuffle", False))
    print_every = int(lms_prm.get("print_every", 1))

    x_ref = np.asarray(x_al, dtype=np.complex128).reshape(-1)
    y0 = np.asarray(y_al, dtype=np.complex128).reshape(-1)

    n0 = min(len(x_ref), len(y0))
    x_ref = x_ref[:n0]
    y0 = y0[:n0]

    phi_x = build_mp_matrix(
        x_ref,
        orders=orders,
        memory_depth=memory_depth,
    )

    n_samples, n_features = phi_x.shape

    # Коэффициенты поправки. Начинаем с нулевого DPD:
    # u = x.
    a = np.zeros(n_features, dtype=np.complex128)

    start_idx = max(0, memory_depth - 1)
    idx = np.arange(start_idx, n_samples)

    eps = 1e-12

    nmse_after_hist = []
    corr_nmse_hist = []
    drive_power_hist = []
    gain_hist = []

    nmse0 = nmse_db_gain_aligned(y0, x_ref)

    print("[LMS-FB] Training feedback LMS predistorter")
    print(f"[LMS-FB] orders = {orders}")
    print(f"[LMS-FB] memory_depth = {memory_depth}")
    print(f"[LMS-FB] n_samples = {n_samples}")
    print(f"[LMS-FB] n_coeffs = {n_features}")
    print(f"[LMS-FB] mu = {mu}")
    print(f"[LMS-FB] epochs per feedback iteration = {epochs}")
    print(f"[LMS-FB] feedback iterations = {fb_iters}")
    print(f"[LMS-FB] feedback_beta = {beta}")
    print(f"[LMS-FB] normalized = {normalized}")
    print(f"[LMS-FB] baseline gain-aligned NMSE = {nmse0:.2f} dB")

    for fb in range(1, fb_iters + 1):
        print("")
        print(f"[LMS-FB] feedback iteration {fb}/{fb_iters}")

        # -------------------------------------------------
        # 1. Текущий предыскаженный сигнал
        # -------------------------------------------------
        correction = phi_x @ a
        u = x_ref + correction

        u, p_ref, p_u_before = normalize_drive_rms(x_ref, u)

        drive_power_hist.append(float(10.0 * np.log10(p_u_before + 1e-15)))

        print(
            f"[LMS-FB] drive RMS before norm = "
            f"{10.0 * np.log10(p_u_before + 1e-15):.2f} dB"
        )

        # -------------------------------------------------
        # 2. Прогон через PA
        # -------------------------------------------------
        y = pa_fn(u)

        n = min(len(x_ref), len(u), len(y))
        x_i = x_ref[:n]
        u_i = u[:n]
        y_i = y[:n]
        phi_i = phi_x[:n, :]

        # -------------------------------------------------
        # 3. Gain alignment выхода PA
        # -------------------------------------------------
        G = estimate_complex_gain(x_i[start_idx:], y_i[start_idx:])
        gain_hist.append(G)

        y_tilde = y_i / (G + 1e-15)

        # -------------------------------------------------
        # 4. Системная ошибка после PA
        # -------------------------------------------------
        e_sys = x_i - y_tilde

        nmse_after = nmse_db_gain_aligned(
            y_i[start_idx:],
            x_i[start_idx:],
        )

        nmse_after_hist.append(float(nmse_after))

        print(f"[LMS-FB] After-PA gain-aligned NMSE = " f"{nmse_after:.2f} dB")

        # -------------------------------------------------
        # 5. Целевая поправка
        #
        # current_correction учитывает RMS-нормировку drive-сигнала,
        # поэтому берем именно u_i - x_i, а не старое phi_x @ a.
        # -------------------------------------------------
        current_correction = u_i - x_i
        c_target = current_correction + beta * e_sys

        # -------------------------------------------------
        # 6. LMS/NLMS обучение:
        #       Phi_x @ a ≈ c_target
        # -------------------------------------------------
        for epoch in range(1, epochs + 1):
            if shuffle:
                np.random.shuffle(idx)

            for k in idx:
                phi_n = phi_i[k, :]
                c_hat_n = phi_n @ a
                e_n = c_target[k] - c_hat_n

                if normalized:
                    step = mu / (np.vdot(phi_n, phi_n).real + eps)
                else:
                    step = mu

                a = a + step * np.conj(phi_n) * e_n

                if not np.all(np.isfinite(a)):
                    raise FloatingPointError(
                        f"LMS feedback diverged at fb_iter={fb}, "
                        f"epoch={epoch}, sample={k}. "
                        f"Try smaller mu or feedback_beta."
                    )

            if (epoch == 1) or (epoch % print_every == 0) or (epoch == epochs):
                c_hat = phi_i @ a

                corr_nmse = nmse_db(
                    c_hat[start_idx:],
                    c_target[start_idx:],
                )

                coef_norm = np.linalg.norm(a)
                corr_nmse_hist.append(float(corr_nmse))

                print(
                    f"  [LMS-FB] epoch {epoch:4d}/{epochs} | "
                    f"corr NMSE={corr_nmse:.2f} dB | "
                    f"coef_norm={coef_norm:.3e}"
                )

    # -------------------------------------------------
    # Финальный DPD
    # -------------------------------------------------
    x_dpd = x_ref + phi_x @ a
    x_dpd, p_ref, p_u_before = normalize_drive_rms(x_ref, x_dpd)

    print(
        f"[LMS-FB] final drive RMS before norm = "
        f"{10.0 * np.log10(p_u_before + 1e-15):.2f} dB"
    )

    model = {
        "a": a,
        "orders": orders,
        "memory_depth": memory_depth,
        "mu": mu,
        "epochs": epochs,
        "ila_iters": fb_iters,
        "feedback_beta": beta,
        "normalized": normalized,
        "shuffle": shuffle,
        "G_hist": np.asarray(gain_hist),
        "nmse_after_hist_db": np.asarray(nmse_after_hist, dtype=float),
        "corr_nmse_hist_db": np.asarray(corr_nmse_hist, dtype=float),
        "drive_power_before_norm_db": np.asarray(drive_power_hist, dtype=float),
        "kind": "lms_feedback_mp",
    }

    return x_dpd.astype(np.complex128), model
