import numpy as np
from typing import Optional


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


def nmse_db(y_hat: np.ndarray, y_ref: np.ndarray):
    """NMSE без дополнительного gain alignment."""
    y_hat = np.asarray(y_hat, dtype=np.complex128).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=np.complex128).reshape(-1)

    n = min(len(y_hat), len(y_ref))
    y_hat = y_hat[:n]
    y_ref = y_ref[:n]

    num = np.mean(np.abs(y_hat - y_ref) ** 2) + 1e-15
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
    """RMS-нормировка drive-сигнала u к уровню исходного сигнала x_ref."""
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    u = np.asarray(u, dtype=np.complex128).reshape(-1)

    n = min(len(x_ref), len(u))
    x_ref = x_ref[:n]
    u = u[:n]

    p_ref = np.mean(np.abs(x_ref) ** 2) + 1e-15
    p_u = np.mean(np.abs(u) ** 2) + 1e-15

    u_norm = u * np.sqrt(p_ref / p_u)

    return u_norm, p_ref, p_u


def estimate_integer_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 20):
    """
    Оценка целочисленной задержки y относительно x.

    lag > 0 означает: y задержан относительно x.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    best_lag = 0
    best_metric = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xs = x[:-lag]
            ys = y[lag:]
        elif lag < 0:
            xs = x[-lag:]
            ys = y[:lag]
        else:
            xs = x
            ys = y

        if len(xs) < 8:
            continue

        metric = np.abs(np.vdot(xs, ys))

        if metric > best_metric:
            best_metric = metric
            best_lag = lag

    return int(best_lag)


def align_pair_by_lag(x: np.ndarray, y: np.ndarray, lag: int):
    """
    Выравнивает пару x, y по заданному lag.

    lag > 0: y задержан относительно x.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if lag > 0:
        return x[:-lag], y[lag:]

    if lag < 0:
        s = -lag
        return x[s:], y[:-s]

    return x, y


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


def build_mp_vector_from_history(
    x_hist: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Формирует один MP-вектор признаков для текущего отсчета.

    x_hist[-1] — текущий отсчет x[n].
    x_hist[-2] — x[n-1], и т.д.

    Порядок признаков такой же, как в build_mp_matrix:
      m = 0: p = 1,3,5...
      m = 1: p = 1,3,5...
      ...
    """
    x_hist = np.asarray(x_hist, dtype=np.complex128).reshape(-1)

    if len(x_hist) < memory_depth:
        pad = np.zeros(memory_depth - len(x_hist), dtype=np.complex128)
        x_hist = np.concatenate([pad, x_hist])

    phi = []

    for m in range(memory_depth):
        z = x_hist[-1 - m]
        az = np.abs(z)

        for p in orders:
            phi.append(z * (az ** (p - 1)))

    return np.asarray(phi, dtype=np.complex128)


def apply_mp_predistorter(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Применение MP-модели:
        correction[n] = Phi_x[n] @ a
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


def apply_mp_feedback_predistorter(
    x: np.ndarray,
    a: np.ndarray,
    orders=(1, 3, 5),
    memory_depth: int = 3,
):
    """
    Residual MP-предистортер:
        u[n] = x[n] + Phi_x[n] @ a
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    correction = apply_mp_predistorter(
        x,
        a,
        orders=orders,
        memory_depth=memory_depth,
    )

    return x + correction


def _run_pa_on_context(pa_fn, context_u: np.ndarray):
    """
    Запускает PA на коротком контексте.
    Возвращает выход той же длины.
    """
    context_u = np.asarray(context_u, dtype=np.complex128).reshape(-1)

    y = pa_fn(context_u)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    n = min(len(context_u), len(y))
    return y[:n]


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
    LMS/NLMS-оценка коэффициентов postdistorter-а:
        x_hat[n] = Phi_y[n] @ a

    Оставлена для совместимости.
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
    lms_prm: Optional[dict] = None,
):
    """
    Online / windowed MP-LMS predistorter with feedback.

    Предистортер:
        u[n] = x[n] + Phi_x[n] @ a

    Обновление:
        a <- a + mu/(eps + ||phi||^2) * conj(phi) * e[n]

    Ошибка:
        e[n] = x[n] - y[n]/G

    Главный фикс относительно предыдущей версии:
      - x_al/y_al уже выровнены ДО входа в эту функцию;
      - поэтому delay нельзя оценивать по x_al/y_al;
      - delay и G_global оцениваются по сырому выходу pa_fn(x_al),
        потому что в цикле обучения pa_fn(u_ctx) также возвращает сырой выход.
    """
    if lms_prm is None:
        lms_prm = {}

    orders = tuple(lms_prm.get("orders", (1, 3, 5)))
    memory_depth = int(lms_prm.get("memory_depth", 5))

    mu = float(lms_prm.get("mu", 1e-4))
    epochs = int(lms_prm.get("epochs", 1))

    block_size = int(lms_prm.get("block_size", 1))
    if block_size <= 0:
        block_size = 1

    left_context = int(
        lms_prm.get(
            "context_len",
            max(4 * memory_depth, 32),
        )
    )

    right_context = int(
        lms_prm.get(
            "right_context",
            max(4 * memory_depth, 32),
        )
    )

    if left_context < memory_depth:
        left_context = memory_depth

    if right_context < 0:
        right_context = 0

    normalized = bool(lms_prm.get("normalized", True))
    feedback_gain = float(lms_prm.get("feedback_gain", 1.0))
    update_sign = float(lms_prm.get("update_sign", 1.0))

    delay = lms_prm.get("delay", None)
    max_lag = int(lms_prm.get("max_lag", 20))

    use_gain = bool(lms_prm.get("use_gain", True))
    use_block_gain = bool(lms_prm.get("use_block_gain", False))
    gain_ref = lms_prm.get("gain_ref", "x")

    power_constraint = bool(lms_prm.get("power_constraint", True))
    keep_best = bool(lms_prm.get("keep_best", True))

    print_every = int(lms_prm.get("print_every", 10000))
    eval_every = int(lms_prm.get("eval_every", 5000))
    max_eval_len = int(lms_prm.get("max_eval_len", 40000))

    coef_leak = float(lms_prm.get("coef_leak", 0.0))
    max_coef_norm = lms_prm.get("max_coef_norm", None)

    min_amp_ratio = float(lms_prm.get("min_amp_ratio", 0.0))

    x_ref = np.asarray(x_al, dtype=np.complex128).reshape(-1)
    y0_aligned = np.asarray(y_al, dtype=np.complex128).reshape(-1)

    n0 = min(len(x_ref), len(y0_aligned))
    x_ref = x_ref[:n0]
    y0_aligned = y0_aligned[:n0]

    n_samples = len(x_ref)
    n_features = len(orders) * memory_depth

    # -------------------------------------------------
    # ВАЖНЫЙ ФИКС: задержка и gain считаются по RAW PA output.
    # y0_aligned уже выровнен, поэтому по нему delay оценивать нельзя.
    # -------------------------------------------------
    y_identity_raw = pa_fn(x_ref)
    y_identity_raw = np.asarray(y_identity_raw, dtype=np.complex128).reshape(-1)
    y_identity_raw = y_identity_raw[:n_samples]

    if delay is None:
        delay = estimate_integer_lag(
            x_ref,
            y_identity_raw,
            max_lag=max_lag,
        )

    delay = int(delay)

    x_g, y_g = align_pair_by_lag(x_ref, y_identity_raw, delay)
    G_global = estimate_complex_gain(x_g, y_g)

    # Residual MP: изначально DPD = identity, поправка Phi*a = 0.
    a = np.zeros(n_features, dtype=np.complex128)

    best_a = a.copy()
    best_nmse = np.inf

    eps = 1e-12
    min_amp = min_amp_ratio * (np.max(np.abs(x_ref)) + eps)

    sample_hist = []
    nmse_after_hist = []
    coef_norm_hist = []
    update_count_hist = []

    baseline_nmse = nmse_db_gain_aligned(y0_aligned, x_ref)
    raw_identity_nmse = nmse_db_gain_aligned(y_identity_raw, x_ref)
    aligned_identity_nmse = nmse_db_gain_aligned(y_g, x_g)

    print("[LMS-MP-ONLINE] Training online/windowed MP feedback LMS predistorter")
    print(f"[LMS-MP-ONLINE] orders = {orders}")
    print(f"[LMS-MP-ONLINE] memory_depth = {memory_depth}")
    print(f"[LMS-MP-ONLINE] n_samples = {n_samples}")
    print(f"[LMS-MP-ONLINE] n_coeffs = {n_features}")
    print(f"[LMS-MP-ONLINE] mu = {mu}")
    print(f"[LMS-MP-ONLINE] epochs = {epochs}")
    print(f"[LMS-MP-ONLINE] block_size = {block_size}")
    print(f"[LMS-MP-ONLINE] left_context = {left_context}")
    print(f"[LMS-MP-ONLINE] right_context = {right_context}")
    print(f"[LMS-MP-ONLINE] normalized = {normalized}")
    print(f"[LMS-MP-ONLINE] feedback_gain = {feedback_gain}")
    print(f"[LMS-MP-ONLINE] update_sign = {update_sign}")
    print(f"[LMS-MP-ONLINE] delay = {delay}")
    print(f"[LMS-MP-ONLINE] use_gain = {use_gain}")
    print(f"[LMS-MP-ONLINE] use_block_gain = {use_block_gain}")
    print(f"[LMS-MP-ONLINE] gain_ref = {gain_ref}")
    print(f"[LMS-MP-ONLINE] power_constraint = {power_constraint}")
    print(f"[LMS-MP-ONLINE] coef_leak = {coef_leak}")
    print(f"[LMS-MP-ONLINE] max_coef_norm = {max_coef_norm}")
    print(f"[LMS-MP-ONLINE] min_amp_ratio = {min_amp_ratio}")
    print(f"[LMS-MP-ONLINE] |G_global| = {np.abs(G_global):.6g}")
    print(f"[LMS-MP-ONLINE] angle(G_global) = {np.angle(G_global, deg=True):.2f} deg")
    print(f"[LMS-MP-ONLINE] baseline aligned-pair NMSE = {baseline_nmse:.2f} dB")
    print(f"[LMS-MP-ONLINE] identity raw NMSE = {raw_identity_nmse:.2f} dB")
    print(f"[LMS-MP-ONLINE] identity aligned NMSE = {aligned_identity_nmse:.2f} dB")

    def make_u_from_x(x_segment, a_current):
        phi = build_mp_matrix(
            x_segment,
            orders=orders,
            memory_depth=memory_depth,
        )
        return x_segment + phi @ a_current

    def evaluate_current_model(tag_sample):
        nonlocal best_a, best_nmse

        n_eval = min(max_eval_len, n_samples)
        x_eval = x_ref[:n_eval]

        u_eval = make_u_from_x(x_eval, a)

        if power_constraint:
            u_eval, _, _ = normalize_drive_rms(x_eval, u_eval)

        y_eval_raw = pa_fn(u_eval)
        x_ev, y_ev = align_pair_by_lag(x_eval, y_eval_raw, delay)

        if use_gain:
            G_ev = estimate_complex_gain(x_ev, y_ev)
            y_ev = y_ev / (G_ev + 1e-15)

        start_idx = max(0, memory_depth - 1)

        if len(x_ev) <= start_idx:
            return

        nmse_val = nmse_db(
            y_ev[start_idx:],
            x_ev[start_idx:],
        )

        sample_hist.append(int(tag_sample))
        nmse_after_hist.append(float(nmse_val))
        coef_norm_hist.append(float(np.linalg.norm(a)))

        if nmse_val < best_nmse:
            best_nmse = nmse_val
            best_a = a.copy()

        print(
            f"[LMS-MP-ONLINE] sample={tag_sample:8d} | "
            f"eval NMSE={nmse_val:.2f} dB | "
            f"best={best_nmse:.2f} dB | "
            f"coef_norm={np.linalg.norm(a):.3e}"
        )

    total_updates = 0
    skipped_low_amp = 0
    start_idx = max(0, memory_depth - 1)

    for epoch in range(1, epochs + 1):
        print("")
        print(f"[LMS-MP-ONLINE] epoch {epoch}/{epochs}")

        n = start_idx

        while n < n_samples:
            blk_start = n
            blk_end = min(n + block_size, n_samples)

            ctx_start = max(0, blk_start - left_context)
            ctx_end = min(n_samples, blk_end + right_context)

            x_ctx = x_ref[ctx_start:ctx_end]

            # PA считается на окне, чтобы модель с памятью видела соседние отсчеты.
            u_ctx = make_u_from_x(x_ctx, a)
            y_ctx = _run_pa_on_context(pa_fn, u_ctx)

            valid0 = blk_start - ctx_start
            valid1 = blk_end - ctx_start

            G_block = G_global
            if use_block_gain:
                y0_idx = valid0 + delay
                y1_idx = valid1 + delay

                if y0_idx >= 0 and y1_idx <= len(y_ctx):
                    y_valid = y_ctx[y0_idx:y1_idx]
                    x_valid = x_ref[blk_start:blk_end]
                    u_valid = u_ctx[valid0:valid1]

                    if len(y_valid) == len(x_valid) and len(y_valid) > 1:
                        if gain_ref == "u":
                            G_block = estimate_complex_gain(u_valid, y_valid)
                        else:
                            G_block = estimate_complex_gain(x_valid, y_valid)

            updates_in_block = 0

            for local_idx in range(valid0, valid1):
                global_idx = ctx_start + local_idx

                if global_idx < start_idx:
                    continue

                if np.abs(x_ref[global_idx]) < min_amp:
                    skipped_low_amp += 1
                    continue

                y_idx = local_idx + delay

                if y_idx < 0 or y_idx >= len(y_ctx):
                    continue

                hist_start = max(0, global_idx - memory_depth + 1)
                x_hist = x_ref[hist_start : global_idx + 1]

                phi_n = build_mp_vector_from_history(
                    x_hist,
                    orders=orders,
                    memory_depth=memory_depth,
                )

                y_n = y_ctx[y_idx]

                if use_gain:
                    y_cmp = y_n / (G_block + 1e-15)
                else:
                    y_cmp = y_n

                e_n = x_ref[global_idx] - y_cmp
                e_n = update_sign * feedback_gain * e_n

                if normalized:
                    step = mu / (np.vdot(phi_n, phi_n).real + eps)
                else:
                    step = mu

                if coef_leak > 0:
                    a = (1.0 - coef_leak) * a

                a = a + step * np.conj(phi_n) * e_n

                if max_coef_norm is not None:
                    norm_a = np.linalg.norm(a)
                    if norm_a > max_coef_norm:
                        a = a * (float(max_coef_norm) / (norm_a + 1e-15))

                updates_in_block += 1
                total_updates += 1

                if not np.all(np.isfinite(a)):
                    raise FloatingPointError(
                        f"Online MP-LMS diverged at epoch={epoch}, "
                        f"sample={global_idx}. Try smaller mu, "
                        f"feedback_gain or update_sign=-1."
                    )

            update_count_hist.append(updates_in_block)

            processed = blk_end + (epoch - 1) * n_samples

            if print_every > 0 and processed % print_every == 0:
                print(
                    f"[LMS-MP-ONLINE] processed={processed}, "
                    f"updates={total_updates}, "
                    f"skipped_low_amp={skipped_low_amp}, "
                    f"coef_norm={np.linalg.norm(a):.3e}"
                )

            if eval_every > 0 and processed % eval_every == 0:
                evaluate_current_model(processed)

            n = blk_end

        evaluate_current_model(epoch * n_samples)

    if keep_best and np.isfinite(best_nmse):
        a_final = best_a
    else:
        a_final = a

    x_dpd = make_u_from_x(x_ref, a_final)

    if power_constraint:
        x_dpd, _, p_u_before = normalize_drive_rms(x_ref, x_dpd)
    else:
        p_u_before = np.mean(np.abs(x_dpd) ** 2) + 1e-15

    print("")
    print(
        f"[LMS-MP-ONLINE] final drive RMS before norm = "
        f"{10.0 * np.log10(p_u_before + 1e-15):.2f} dB"
    )
    print(f"[LMS-MP-ONLINE] total LMS updates = {total_updates}")
    print(f"[LMS-MP-ONLINE] skipped low-amplitude samples = {skipped_low_amp}")
    print(f"[LMS-MP-ONLINE] best eval NMSE = {best_nmse:.2f} dB")

    model = {
        "a": a_final,
        "orders": orders,
        "memory_depth": memory_depth,
        "mu": mu,
        "epochs": epochs,
        "block_size": block_size,
        "context_len": left_context,
        "left_context": left_context,
        "right_context": right_context,
        "delay": delay,
        "G_global": G_global,
        "use_gain": use_gain,
        "use_block_gain": use_block_gain,
        "gain_ref": gain_ref,
        "normalized": normalized,
        "feedback_gain": feedback_gain,
        "update_sign": update_sign,
        "power_constraint": power_constraint,
        "coef_leak": coef_leak,
        "max_coef_norm": max_coef_norm,
        "min_amp_ratio": min_amp_ratio,
        "best_nmse_after_db": best_nmse,
        "sample_hist": np.asarray(sample_hist, dtype=int),
        "nmse_after_hist_db": np.asarray(nmse_after_hist, dtype=float),
        "coef_norm_hist": np.asarray(coef_norm_hist, dtype=float),
        "update_count_hist": np.asarray(update_count_hist, dtype=int),
        "total_updates": int(total_updates),
        "skipped_low_amp": int(skipped_low_amp),
        "kind": "lms_mp_online_feedback_delay_fixed",
    }

    return x_dpd.astype(np.complex128), model
