# lms_alg.py
import numpy as np
from ls_alg import build_poly_matrix  # используем ТО ЖЕ, что и LS


def lms_postdistorter_coeffs(
    y: np.ndarray,
    x: np.ndarray,
    orders=(1, 3, 5),
    mu: float = 1e-6,
    epochs: int = 1,
    w0: np.ndarray | None = None,
):
    """
    ILA (postdistorter) LMS:
      Solve iteratively:  min ||Phi(y) a - x||^2

    Phi(y) строится ровно так же, как в LS (через build_poly_matrix).
    """
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    Phi = build_poly_matrix(y, orders=orders)  # (N,P), как в LS
    N, P = Phi.shape

    a = np.zeros(P, dtype=np.complex128) if w0 is None else np.asarray(w0, dtype=np.complex128).reshape(-1)

    for _ in range(epochs):
        for n in range(N):
            phi = Phi[n]                
            x_hat = np.vdot(a, phi)      
            e = x[n] - x_hat
            # print(f"n={n}, e={e:.3e}, x[n]={x[n]:.3e}, x_hat={x_hat:.3e}")
            a = a + mu * phi * np.conj(e)

    return a