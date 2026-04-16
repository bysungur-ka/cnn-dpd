import numpy as np


def _iir_filter_complex(x, b, a):
    x = np.asarray(x, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    a = np.asarray(a, dtype=np.complex128)

    if a.size == 0 or np.abs(a[0]) < 1e-15:
        raise ValueError("IIR denominator a[0] must be nonzero.")

    if np.abs(a[0] - 1.0) > 1e-15:
        b = b / a[0]
        a = a / a[0]

    n_samples = len(x)
    nb = len(b)
    na = len(a)

    y = np.zeros(n_samples, dtype=np.complex128)

    for n in range(n_samples):
        acc = 0.0 + 0.0j

        for k in range(nb):
            if n - k >= 0:
                acc += b[k] * x[n - k]

        for m in range(1, na):
            if n - m >= 0:
                acc -= a[m] * y[n - m]

        y[n] = acc

    return y


def _shift_signal(x, shift):
    """
    Возвращает x[n - shift] с zero-padding.

    shift > 0 : задержка  -> x[n-shift]
    shift = 0 : без сдвига
    shift < 0 : опережение -> x[n+|shift|]
    """
    x = np.asarray(x, dtype=np.complex128)
    n_samples = len(x)
    y = np.zeros_like(x)

    if shift == 0:
        y[:] = x
    elif shift > 0:
        if shift < n_samples:
            y[shift:] = x[: n_samples - shift]
    else:
        lead = -shift
        if lead < n_samples:
            y[: n_samples - lead] = x[lead:]

    return y


def _get_complex_dict(user_dict):
    if user_dict is None:
        return None
    return {tuple(map(int, k)): np.complex128(v) for k, v in user_dict.items()}


def amp_model(prm, x):
    """
    PA model with four modes:

    1) pa_mode = 'gmp_like'
       Упрощённая GMP-like модель:
       - main memory polynomial branch
       - один lagging envelope cross-term

    2) pa_mode = 'gmp'
       Структура GMP по Morgan:
       - aligned terms
       - lagging envelope cross-terms
       - leading envelope cross-terms (опционально)

    3) pa_mode = 'mp'
       Memory Polynomial model:
           y[n] = sum_{k odd} sum_q b_{kq} x[n-q] |x[n-q]|^(k-1)

    4) pa_mode = 'iir'
       Полиномиальная нелинейность + IIR память
    """
    x = np.asarray(x, dtype=np.complex128)
    mode = str(prm.get("pa_mode", "gmp_like")).lower()

    # Общие коэффициенты, которые можно переиспользовать
    c1 = 14.974 + 1j * 0.0519
    c3 = -23.0954 + 1j * 4.968
    c5 = 21.3936 + 1j * 0.4305

    # -------------------------------------------------
    # 1) GMP-like mode (старый твой вариант)
    # -------------------------------------------------
    if mode == "gmp_like":
        alpha = float(prm.get("pa_alpha", 1.0))
        memory_depth = int(prm.get("pa_memory", 3))
        env_lag_k = int(prm.get("gmp_k", 2))
        beta = float(prm.get("gmp_beta", 0.15))
        mem_decay = float(prm.get("mem_decay", 0.7))

        n_samples = len(x)
        y = np.zeros_like(x)
        env2 = np.abs(x) ** 2

        for m in range(memory_depth):
            w = mem_decay**m
            xm = _shift_signal(x, m)

            y += w * xm * (
                c1
                + alpha * c3 * np.abs(xm) ** 2
                + (alpha**2) * c5 * np.abs(xm) ** 4
            )

            if env_lag_k > 0:
                env_lag = np.abs(_shift_signal(x, m + env_lag_k)) ** 2
                y += w * (beta * alpha) * xm * env_lag

        return y

    # -------------------------------------------------
    # 2) Morgan-style GMP
    # -------------------------------------------------
    elif mode == "gmp":
        alpha = float(prm.get("pa_alpha", 1.0))
        beta = float(prm.get("gmp_beta", 0.15))
        mem_decay = float(prm.get("mem_decay", 0.7))

        # ----- Index sets / structure -----
        aligned_orders = list(prm.get("gmp_aligned_orders", [1, 3, 5]))
        aligned_memory = int(prm.get("gmp_aligned_memory", prm.get("pa_memory", 3)))

        lag_orders = list(prm.get("gmp_lag_orders", [3]))
        lag_memory = int(prm.get("gmp_lag_memory", prm.get("pa_memory", 3)))
        lag_env_delays = list(prm.get("gmp_lag_env_delays", [int(prm.get("gmp_k", 2))]))

        lead_orders = list(prm.get("gmp_lead_orders", []))
        lead_memory = int(prm.get("gmp_lead_memory", prm.get("pa_memory", 3)))
        lead_env_delays = list(prm.get("gmp_lead_env_delays", []))

        # ----- Optional user coefficients -----
        # dict formats:
        #   gmp_a_coeffs[(p, m)] = coeff
        #   gmp_b_coeffs[(p, m, l)] = coeff
        #   gmp_c_coeffs[(p, m, l)] = coeff
        a_user = _get_complex_dict(prm.get("gmp_a_coeffs", None))
        b_user = _get_complex_dict(prm.get("gmp_b_coeffs", None))
        c_user = _get_complex_dict(prm.get("gmp_c_coeffs", None))

        # ----- Defaults close to current gmp_like behavior -----
        aligned_base = {
            1: c1,
            3: alpha * c3,
            5: (alpha**2) * c5,
        }

        y = np.zeros_like(x)

        # ----- Aligned terms -----
        for p in aligned_orders:
            for m in range(aligned_memory):
                xm = _shift_signal(x, m)

                if a_user is None:
                    coeff = (mem_decay**m) * aligned_base.get(p, 0.0 + 0.0j)
                else:
                    coeff = a_user.get((p, m), 0.0 + 0.0j)

                if coeff == 0:
                    continue

                y += coeff * xm * (np.abs(xm) ** (p - 1))

        # ----- Lagging envelope cross-terms -----
        for p in lag_orders:
            for m in range(lag_memory):
                xm = _shift_signal(x, m)

                for l in lag_env_delays:
                    x_env = _shift_signal(x, m + l)

                    if b_user is None:
                        # по умолчанию оставляем аналог старой модели:
                        # только p=3 имеет ненулевой cross-term
                        if p == 3:
                            coeff = (mem_decay**m) * (beta * alpha)
                        else:
                            coeff = 0.0 + 0.0j
                    else:
                        coeff = b_user.get((p, m, l), 0.0 + 0.0j)

                    if coeff == 0:
                        continue

                    y += coeff * xm * (np.abs(x_env) ** (p - 1))

        # ----- Leading envelope cross-terms -----
        for p in lead_orders:
            for m in range(lead_memory):
                xm = _shift_signal(x, m)

                for l in lead_env_delays:
                    x_env = _shift_signal(x, m - l)  # x[n-m+l]

                    if c_user is None:
                        # по умолчанию leading выключен, если coeffs не заданы явно
                        coeff = 0.0 + 0.0j
                    else:
                        coeff = c_user.get((p, m, l), 0.0 + 0.0j)

                    if coeff == 0:
                        continue

                    y += coeff * xm * (np.abs(x_env) ** (p - 1))

        return y

    # -------------------------------------------------
    # 3) MP mode
    # -------------------------------------------------
    elif mode == "mp":
        # По умолчанию: коэффициенты из Ding, Example 3.5
        orders = list(prm.get("mp_orders", [1, 3, 5]))
        memory_depth = int(prm.get("mp_memory_depth", 3))
        mp_alpha = float(prm.get("mp_alpha", 1.0))

        # Если пользователь передал свои коэффициенты:
        # формат: coeffs[(k, q)] = complex
        user_coeffs = prm.get("mp_coeffs", None)

        if user_coeffs is None:
            coeffs = {
                (1, 0): 1.0513 + 0.0904j,
                (3, 0): -0.0542 - 0.2900j,
                (5, 0): -0.9657 - 0.7028j,
                (1, 1): -0.0680 - 0.0023j,
                (3, 1): 0.2234 + 0.2317j,
                (5, 1): -0.2451 - 0.3735j,
                (1, 2): 0.0289 - 0.0054j,
                (3, 2): -0.0621 - 0.0932j,
                (5, 2): 0.1229 + 0.1508j,
            }
        else:
            coeffs = {
                (int(k), int(q)): np.complex128(v)
                for (k, q), v in user_coeffs.items()
            }

        y = np.zeros_like(x)

        for q in range(memory_depth):
            xq = _shift_signal(x, q)
            abs_xq = np.abs(xq)

            for p in orders:
                bkq = coeffs.get((p, q), 0.0 + 0.0j)
                if bkq == 0:
                    continue

                if p == 1:
                    scale_p = 1.0
                else:
                    scale_p = mp_alpha ** ((p - 1) // 2)

                y += scale_p * bkq * xq * (abs_xq ** (p - 1))

        return y

    # -------------------------------------------------
    # 4) IIR mode
    # -------------------------------------------------
    elif mode == "iir":
        alpha = float(prm.get("pa_alpha", 1.0))

        z = x * (
            c1
            + alpha * c3 * np.abs(x) ** 2
            + (alpha**2) * c5 * np.abs(x) ** 4
        )

        b = prm.get("pa_b", [0.85, 0.12])
        a = prm.get("pa_a", [1.0, -0.55, 0.16])
        gain = float(prm.get("pa_gain", 1.0))

        y = _iir_filter_complex(z, b=b, a=a)
        y = gain * y
        return y

    else:
        raise ValueError("pa_mode must be 'gmp_like', 'gmp', 'mp' or 'iir'")