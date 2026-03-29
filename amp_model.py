import numpy as np


def _iir_filter_complex(x, b, a):
    """
    Комплексный IIR-фильтр:
        y[n] = sum_k b[k] * x[n-k] - sum_m a[m] * y[n-m],   m >= 1
    """
    x = np.asarray(x, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    a = np.asarray(a, dtype=np.complex128)

    if a.size == 0 or np.abs(a[0]) < 1e-15:
        raise ValueError("IIR denominator a[0] must be nonzero.")

    # Нормировка к a[0] = 1
    if np.abs(a[0] - 1.0) > 1e-15:
        b = b / a[0]
        a = a / a[0]

    N = len(x)
    nb = len(b)
    na = len(a)

    y = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        acc = 0.0 + 0.0j

        # FIR-часть
        for k in range(nb):
            if n - k >= 0:
                acc += b[k] * x[n - k]

        # Рекурсивная часть
        for m in range(1, na):
            if n - m >= 0:
                acc -= a[m] * y[n - m]

        y[n] = acc

    return y


def amp_model(prm, x):
    """
    Модель усилителя мощности с двумя режимами:

    1) pa_mode = 'gmp'  (по умолчанию)
       GMP-lite:
         - main memory polynomial
         - один lagging envelope cross-term

    2) pa_mode = 'iir'
       Полиномиальная нелинейность + IIR-память:
         z[n] = x[n] * (c1 + alpha*c3*|x[n]|^2 + alpha^2*c5*|x[n]|^4)
         y[n] = IIR{ z[n] }
    """

    # Полиномиальные коэффициенты
    c1 = 14.974 + 1j * 0.0519
    c3 = -23.0954 + 1j * 4.968
    c5 = 21.3936 + 1j * 0.4305

    x = np.asarray(x, dtype=np.complex128)
    mode = str(prm.get("pa_mode", "gmp")).lower()
    alpha = float(prm.get("pa_alpha", 1.0))

    # -------------------------------------------------
    # Режим 1: текущая GMP-lite модель
    # -------------------------------------------------
    if mode == "gmp":
        M = int(prm.get("pa_memory", 3))
        k = int(prm.get("gmp_k", 2))
        beta = float(prm.get("gmp_beta", 0.15))
        mem_decay = float(prm.get("mem_decay", 0.7))

        N = len(x)
        y = np.zeros_like(x)

        env2 = np.abs(x) ** 2

        for m in range(M):
            w = mem_decay ** m

            xm = np.zeros_like(x)
            xm[m:] = x[:N - m]

            # Main branch
            y += w * xm * (
                c1
                + alpha * c3 * np.abs(xm) ** 2
                + (alpha ** 2) * c5 * np.abs(xm) ** 4
            )

            # Lagging envelope cross-term
            if k > 0:
                env_lag = np.zeros_like(env2)
                shift = m + k
                if shift < N:
                    env_lag[shift:] = env2[:N - shift]

                y += w * (beta * alpha) * xm * env_lag

        return y

    # -------------------------------------------------
    # Режим 2: нелинейность + IIR-память
    # -------------------------------------------------
    elif mode == "iir":
        # Статическая полиномиальная нелинейность
        z = x * (
            c1
            + alpha * c3 * np.abs(x) ** 2
            + (alpha ** 2) * c5 * np.abs(x) ** 4
        )

        # Коэффициенты IIR-фильтра
        b = prm.get("pa_b", [0.85, 0.12])
        a = prm.get("pa_a", [1.0, -0.55, 0.16])
        gain = float(prm.get("pa_gain", 1.0))

        y = _iir_filter_complex(z, b=b, a=a)
        y = gain * y
        return y

    else:
        raise ValueError("pa_mode must be 'gmp' or 'iir'")