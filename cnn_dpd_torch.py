import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_feature_channels_torch(u: torch.Tensor, mode: str) -> torch.Tensor:
    ur = u.real
    ui = u.imag

    if mode.lower() == "iq":
        X = torch.stack([ur, ui], dim=0)  # (2,N)
        return X.to(torch.float32)

    if mode.lower() == "poly":
        a2 = (ur * ur + ui * ui)  # |u|^2 (real)
        u2r = ur * a2
        u2i = ui * a2
        a4 = a2 * a2
        u4r = ur * a4
        u4i = ui * a4
        X = torch.stack([ur, ui, u2r, u2i, u4r, u4i], dim=0)  # (6,N)
        return X.to(torch.float32)

    raise ValueError('features must be "iq" or "poly"')


def _estimate_complex_gain_ls(x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    xv = x[idx]
    yv = y[idx]
    denom = torch.vdot(xv, xv) + (1e-15 + 0j)
    G = torch.vdot(xv, yv) / denom
    return G


def _gain_aligned_nmse_db_np(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    """
    NMSE between x and gain-aligned y:
    minimize ||x - (y/G)||^2 over complex scalar G
    (equiv to LS y ≈ Gx, then y/G aligned to x)
    """
    x = np.asarray(x, np.complex128)
    y = np.asarray(y, np.complex128)

    xv = x[idx]
    yv = y[idx]

    denom = np.vdot(xv, xv) + 1e-15
    G = np.vdot(xv, yv) / denom

    y_al = yv / (G + 1e-15)
    err = y_al - xv
    nmse = (np.mean(np.abs(err) ** 2) + 1e-15) / (np.mean(np.abs(xv) ** 2) + 1e-15)
    return 10 * np.log10(nmse + 1e-15)


class CNNPostDistorter(nn.Module):
    def __init__(self, C: int, K: int, Ff: int, M1: int, M2: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.K = int(K)

        self.conv = nn.Conv1d(in_channels=C, out_channels=Ff, kernel_size=K, bias=True)
        self.fc1 = nn.Linear(Ff, M1, bias=True)
        self.fc2 = nn.Linear(M1, M2, bias=True)
        self.out = nn.Linear(M2, 2, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X: (B, C, N) for full sequence
               or (B, C, K) for windowed batching
        Output:
            (B, 2, L), where L = N-K+1
            For windowed batching with N=K => L=1
        """
        Z = self.conv(X)             # (B, Ff, L)
        A = F.relu(Z)
        A_t = A.transpose(1, 2)      # (B, L, Ff)
        H1 = F.relu(self.fc1(A_t))
        H2 = F.relu(self.fc2(H1))
        Y = self.out(H2)             # (B, L, 2)
        return Y.transpose(1, 2)     # (B, 2, L)


def _build_window_dataset(
    X_seq: torch.Tensor,    # (C, N)
    Y_target: torch.Tensor, # (2, N)
    K: int,
):
    """
    Формирует датасет окон для обучения постдистортера.

    Returns:
        X_win: (L, C, K),  L = N-K+1
        Y_win: (L, 2)
    """
    # unfold along time axis -> (C, L, K)
    X_win = X_seq.unfold(dimension=1, size=K, step=1).permute(1, 0, 2).contiguous()
    Y_win = Y_target[:, K - 1:].transpose(0, 1).contiguous()
    return X_win, Y_win


def _iter_batch_indices(L: int, batch_size: int, mode: str, device: torch.device):
    """
    Генератор индексов батчей.

    mode:
        - "random": случайные окна с повторениями
        - "contig": contiguous-блоки в случайном порядке
    """
    if batch_size >= L:
        yield torch.arange(L, device=device, dtype=torch.long)
        return

    if mode == "random":
        steps = max(1, int(np.ceil(L / batch_size)))
        for _ in range(steps):
            yield torch.randint(
                low=0,
                high=L,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        return

    if mode == "contig":
        starts = torch.arange(0, L, batch_size, device=device, dtype=torch.long)
        order = torch.randperm(starts.numel(), device=device)
        for idx in order:
            s = int(starts[idx].item())
            e = min(s + batch_size, L)
            yield torch.arange(s, e, device=device, dtype=torch.long)
        return

    raise ValueError('batch_mode must be "random" or "contig"')


@torch.no_grad()
def apply_predistorter_torch(
    model: CNNPostDistorter,
    x: torch.Tensor,
    features: str,
    feat_rms: torch.Tensor,
    device: torch.device,
    power_constraint: bool = True,
) -> torch.Tensor:
    K = int(model.K)
    N = x.numel()
    n_valid = torch.arange(K - 1, N, device=device, dtype=torch.long)

    Xx = _make_feature_channels_torch(x, features).to(device)  # (C,N)
    Xx = Xx / feat_rms

    Y = model(Xx.unsqueeze(0)).squeeze(0)  # (2,L)

    x_dpd = x.clone()
    delta = (Y[0, :] + 1j * Y[1, :]).to(x_dpd.dtype)  # (L,)

    if model.residual:
        x_hat = x[n_valid] + delta
    else:
        x_hat = delta

    x_dpd[n_valid] = x_hat

    if power_constraint:
        p_ref = torch.mean(torch.abs(x[n_valid]) ** 2) + 1e-15
        p_dpd = torch.mean(torch.abs(x_dpd[n_valid]) ** 2) + 1e-15
        x_dpd = x_dpd * torch.sqrt(p_ref / p_dpd)

    return x_dpd


def cnn_dpd_torch(
    sig_clean: np.ndarray,
    sig_distorted: np.ndarray,
    prm: dict,
    pa_fn=None,
):
    cnn_prm = prm.get("cnn", {})
    K_kernel = int(cnn_prm.get("kernel", int(cnn_prm.get("memory", 1))))
    Ff = int(cnn_prm.get("filters", 32))
    M1 = int(cnn_prm.get("M1", 64))
    M2 = int(cnn_prm.get("M2", 64))
    lr = float(cnn_prm.get("lr", 1e-3))
    epochs = int(cnn_prm.get("epochs", 300))
    seed = int(cnn_prm.get("seed", 42))
    print_every = int(cnn_prm.get("print_every", 10))
    features = str(cnn_prm.get("features", "poly"))
    batch_size = int(cnn_prm.get("batch_size", 4096))
    batch_mode = str(cnn_prm.get("batch_mode", "contig")).lower()
    power_constraint = bool(cnn_prm.get("power_constraint", False))
    residual = bool(cnn_prm.get("residual", False))
    ila_iters = int(cnn_prm.get("ila_iters", 3))
    warm_start = bool(cnn_prm.get("warm_start", False))
    betas = cnn_prm.get("adam_betas", (0.9, 0.999))
    weight_decay = float(cnn_prm.get("weight_decay", 0.0))
    grad_clip = float(cnn_prm.get("clip", 0.0))

    dev = str(cnn_prm.get("device", "cuda" if torch.cuda.is_available() else "cpu")).lower()
    device = torch.device(dev)

    sig_clean = np.asarray(sig_clean, dtype=np.complex128)
    sig_distorted = np.asarray(sig_distorted, dtype=np.complex128)
    N = min(len(sig_clean), len(sig_distorted))

    x_np = sig_clean[:N]
    y0_np = sig_distorted[:N]

    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(x_np).to(device)
    y0 = torch.from_numpy(y0_np).to(device)

    # fixed reference x
    x_ref = x

    K = int(K_kernel)
    n_valid = torch.arange(K - 1, N, device=device, dtype=torch.long)
    if int(n_valid.numel()) <= 0:
        raise ValueError("Not enough samples for kernel length K.")
    idx_np = np.arange(K - 1, N, dtype=np.int64)

    # model
    C0 = int(_make_feature_channels_torch(x_ref, features).shape[0])
    model = CNNPostDistorter(C=C0, K=K, Ff=Ff, M1=M1, M2=M2, residual=residual).to(device)

    G_hist = []
    nmse_train_hist = []
    nmse_after_hist = []

    # --- ILA loop ---
    last_nmse_db = None
    feat_rms_last = None

    # u = PA input for current iteration
    u = x_ref.clone()
    y = y0.clone()

    for ila in range(ila_iters):
        # 0) define u_k and y_k
        if ila == 0:
            # iteration 0 uses provided (x_ref -> PA -> y0)
            u = x_ref
            y = y0
        else:
            if pa_fn is None:
                raise ValueError("pa_fn is required for ILA iterations > 1 (need PA feedback).")

            # current predistorter applied to fixed reference -> u_k
            model.eval()
            with torch.no_grad():
                if feat_rms_last is None:
                    # safe fallback: normalize features by x_ref if first time
                    Xref = _make_feature_channels_torch(x_ref, features).to(device)
                    feat_rms_last = torch.sqrt(torch.mean(Xref[:, n_valid] ** 2, dim=1, keepdim=True) + 1e-12)

                u = apply_predistorter_torch(
                    model=model,
                    x=x_ref,
                    features=features,
                    feat_rms=feat_rms_last,
                    device=device,
                    power_constraint=power_constraint,
                )

            # AGC on PA input u (stabilize PA operating point)
            u_np = u.detach().cpu().numpy().astype(np.complex128)
            p_ref = np.mean(np.abs(x_np[idx_np]) ** 2) + 1e-15
            p_in = np.mean(np.abs(u_np[idx_np]) ** 2) + 1e-15
            u_np = u_np * np.sqrt(p_ref / p_in)
            u = torch.from_numpy(u_np[:N]).to(device)

            # PA feedback
            y_np = np.asarray(pa_fn(u_np), dtype=np.complex128)[:N]
            y = torch.from_numpy(y_np).to(device)

        # 1) estimate gain and normalize using (u, y)
        G = _estimate_complex_gain_ls(u, y, n_valid)
        y_tilde = y / (G + (1e-15 + 0j))
        G_hist.append(G.detach().cpu().numpy())

        print(f"\nILA iter {ila+1}/{ila_iters} | train post: y/G -> u | residual={residual} | device={device} | G={G.detach().cpu().numpy():.6g}")

        # 2) features normalization: by current u distribution
        Xu = _make_feature_channels_torch(u, features).to(device)  # (C,N)
        feat_rms = torch.sqrt(torch.mean(Xu[:, n_valid] ** 2, dim=1, keepdim=True) + 1e-12)
        feat_rms_last = feat_rms  # keep for next iteration predistort

        Xy = _make_feature_channels_torch(y_tilde, features).to(device) / feat_rms

        # 3) targets: u (NOT x_ref)
        Y_u = torch.stack([u.real, u.imag], dim=0).to(torch.float32)              # (2,N)
        Y_ytilde = torch.stack([y_tilde.real, y_tilde.imag], dim=0).to(torch.float32)  # (2,N)
        Y_target = (Y_u - Y_ytilde) if residual else Y_u

        # 4) reinit if needed
        if (ila > 0) and (not warm_start):
            model = CNNPostDistorter(C=C0, K=K, Ff=Ff, M1=M1, M2=M2, residual=residual).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)

        # 4.5) build real batch dataset once
        X_train, Y_train = _build_window_dataset(Xy, Y_target, K)
        L = int(X_train.shape[0])

        # 5) train postdistorter (forward per batch)
        for ep in range(epochs):
            model.train()

            for b in _iter_batch_indices(L, batch_size, batch_mode, device):
                Xb = X_train[b]                    # (B,C,K)
                Yb = Y_train[b]                    # (B,2)

                Yh = model(Xb).squeeze(-1)         # (B,2)

                loss = torch.mean((Yh - Yb) ** 2)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                opt.step()

            if (ep == 0) or ((ep + 1) % print_every == 0):
                model.eval()
                with torch.no_grad():
                    Yh = model(Xy.unsqueeze(0)).squeeze(0)  # (2,L)

                    if residual:
                        base = Y_ytilde[:, n_valid]  # (2,L)
                        Yhat = Yh + base
                        Yv = Y_u[:, n_valid]
                    else:
                        Yhat = Yh
                        Yv = Y_u[:, n_valid]

                    err = (Yhat - Yv).to(torch.float32)
                    mse = torch.mean(err ** 2).item()
                    p_ref_t = torch.mean(Yv.to(torch.float32) ** 2).item() + 1e-15
                    nmse = mse / p_ref_t
                    nmse_db = 10 * np.log10(nmse + 1e-15)
                    last_nmse_db = nmse_db

                print(f" Epoch {ep+1:5d}/{epochs} | Train NMSE(u)={nmse_db:.2f} dB")

        if last_nmse_db is not None:
            nmse_train_hist.append(float(last_nmse_db))

        # --- report system-level metric: PA(u_next) vs fixed x_ref ---
        if pa_fn is not None:
            # build u_next = f(x_ref) with updated model
            model.eval()
            with torch.no_grad():
                u_next = apply_predistorter_torch(
                    model=model,
                    x=x_ref,
                    features=features,
                    feat_rms=feat_rms,  # use current normalization
                    device=device,
                    power_constraint=power_constraint,
                )

            u_next_np = u_next.detach().cpu().numpy().astype(np.complex128)

            # AGC on PA input again for fair comparison
            p_ref = np.mean(np.abs(x_np[idx_np]) ** 2) + 1e-15
            p_in = np.mean(np.abs(u_next_np[idx_np]) ** 2) + 1e-15
            u_next_np = u_next_np * np.sqrt(p_ref / p_in)

            y_lin_np = np.asarray(pa_fn(u_next_np), dtype=np.complex128)[:N]
            nmse_after_db = _gain_aligned_nmse_db_np(x_np, y_lin_np, idx_np)
            nmse_after_hist.append(float(nmse_after_db))

            print(f" ILA iter {ila+1}/{ila_iters} | After-PA gain-aligned NMSE(x_ref) = {nmse_after_db:.2f} dB")

    model_dict = {
        "torch_state_dict": model.state_dict(),
        "G_hist": np.asarray(G_hist),
        "feat_rms": feat_rms_last.detach().cpu().numpy() if feat_rms_last is not None else None,
        "features": features,
        "K": K,
        "F": Ff,
        "M1": M1,
        "M2": M2,
        "residual": residual,
        "power_constraint": power_constraint,
        "device": str(device),
        "optimizer": "Adam",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "batch_mode": batch_mode,
        "weight_decay": weight_decay,
        "clip": grad_clip,
        "ila_iters": ila_iters,
        "warm_start": warm_start,
        "nmse_train_hist_db": np.asarray(nmse_train_hist, dtype=float),
        "nmse_after_hist_db": np.asarray(nmse_after_hist, dtype=float),
    }

    # final predistorted output for the caller: u_final = f(x_ref)
    # (use last feat_rms)
    model.eval()
    with torch.no_grad():
        x_dpd_final = apply_predistorter_torch(
            model=model,
            x=x_ref,
            features=features,
            feat_rms=feat_rms_last,
            device=device,
            power_constraint=power_constraint,
        )

    return x_dpd_final.detach().cpu().numpy().astype(np.complex128), model_dict