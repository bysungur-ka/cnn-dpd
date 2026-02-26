# cnn_dpd.py (PyTorch ILA)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_feature_channels_torch(u: torch.Tensor, mode: str) -> torch.Tensor:
    """
    u: complex tensor (N,)
    returns: float tensor (C,N)
    """
    ur = u.real
    ui = u.imag

    if mode.lower() == "iq":
        return torch.stack([ur, ui], dim=0)

    if mode.lower() == "poly":
        a2 = ur * ur + ui * ui
        u2r = ur * a2
        u2i = ui * a2
        a4 = a2 * a2
        u4r = ur * a4
        u4i = ui * a4
        return torch.stack([ur, ui, u2r, u2i, u4r, u4i], dim=0)

    raise ValueError('features must be "iq" or "poly"')


class PostDistorterCNN(nn.Module):
    """
    Proper multi-channel 1D CNN postdistorter:
      input:  (B, C, N)
      output: (B, 2, N)  -> estimated x IQ
    Causal Conv1D: left padding K-1 then crop.
    """
    def __init__(self, C: int, Fch: int, K: int, M1: int, M2: int):
        super().__init__()
        self.K = K
        self.conv = nn.Conv1d(C, Fch, kernel_size=K, padding=K-1, bias=True)
        self.fc1 = nn.Conv1d(Fch, M1, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(M1, M2, kernel_size=1, bias=True)
        self.out = nn.Conv1d(M2, 2, kernel_size=1, bias=True)

        # Optional: small init on final to start near 0 output
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.conv(X)
        z = z[..., :X.shape[-1]]  # crop to length N (causal)
        z = F.relu(z)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        y = self.out(z)
        return y


def cnn_dpd(sig_clean: np.ndarray, sig_distorted: np.ndarray, prm: dict):
    """
    ILA (Indirect Learning Architecture), PyTorch implementation.

    Train postdistorter:
        g_theta(y) ≈ x
    Then apply as predistorter:
        x_dpd = g_theta(x)

    Inputs MUST be aligned pairs:
        sig_clean      = x (reference)
        sig_distorted  = y (PA output)
    """

    cnn_prm = prm.get("cnn", {})
    K = int(cnn_prm.get("kernel", cnn_prm.get("memory", 5)))
    Fch = int(cnn_prm.get("filters", 8))
    M1 = int(cnn_prm.get("M1", 64))
    M2 = int(cnn_prm.get("M2", 64))
    epochs = int(cnn_prm.get("epochs", 1000))
    lr = float(cnn_prm.get("lr", 1e-3))
    print_every = int(cnn_prm.get("print_every", 50))
    features = str(cnn_prm.get("features", "poly"))  # iq/poly
    weight_decay = float(cnn_prm.get("weight_decay", 0.0))
    grad_clip = float(cnn_prm.get("grad_clip", 1.0))
    device_req = str(cnn_prm.get("device", "auto"))

    # device
    if device_req == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_req == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy -> torch
    x_np = np.asarray(sig_clean, dtype=np.complex128)
    y_np = np.asarray(sig_distorted, dtype=np.complex128)
    N = min(len(x_np), len(y_np))
    x_np = x_np[:N]
    y_np = y_np[:N]

    x = torch.tensor(x_np, dtype=torch.complex64, device=device)
    y = torch.tensor(y_np, dtype=torch.complex64, device=device)

    # valid indices
    n_valid = torch.arange(K - 1, N, device=device)
    if n_valid.numel() <= 0:
        raise ValueError("Not enough samples for given K.")

    # build features from y (postdistorter input)
    Xy = _make_feature_channels_torch(y, features)  # (C,N)
    C = Xy.shape[0]

    # per-channel RMS normalization on valid region (crucial)
    feat_rms = torch.sqrt(torch.mean(Xy[:, n_valid] ** 2, dim=1, keepdim=True) + 1e-12)  # (C,1)
    Xy = Xy / feat_rms

    # targets: x IQ
    Yt = torch.stack([x.real, x.imag], dim=0)  # (2,N)

    # model + optimizer
    net = PostDistorterCNN(C=C, Fch=Fch, K=K, M1=M1, M2=M2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Training CNN ILA (PyTorch) on {device}, features={features}, C={C}, K={K}, F={Fch}...")

    for ep in range(epochs):
        net.train()
        opt.zero_grad()

        # forward
        X = Xy.unsqueeze(0)            # (1,C,N)
        Yh = net(X).squeeze(0)         # (2,N)

        # loss on valid only
        err = Yh[:, n_valid] - Yt[:, n_valid]
        loss = torch.mean(err ** 2)    # mean over 2*N_valid

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        if ep == 0 or (ep + 1) % print_every == 0:
            with torch.no_grad():
                mse = loss
                p_ref = torch.mean(Yt[:, n_valid] ** 2) + 1e-12
                nmse = mse / p_ref
                print(f"Epoch {ep+1:4d} | MSE={10*torch.log10(mse+1e-12).item():.2f} dB | NMSE={10*torch.log10(nmse+1e-12).item():.2f} dB")

    print("Generating predistorted signal (apply postdistorter as predistorter)...")

    # apply g to x (IMPORTANT: same normalization feat_rms, but features built from x)
    net.eval()
    with torch.no_grad():
        Xx = _make_feature_channels_torch(x, features) / feat_rms  # (C,N)
        Yd = net(Xx.unsqueeze(0)).squeeze(0)                       # (2,N)
        x_dpd = Yd[0, :] + 1j * Yd[1, :]
        x_dpd[:K-1] = x[:K-1]  # protect prefix

    x_dpd_np = x_dpd.detach().cpu().numpy().astype(np.complex128)

    model = {
        "torch": True,
        "device": str(device),
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "feat_rms": feat_rms.detach().cpu().numpy(),
        "features": features,
        "K": K, "F": Fch, "M1": M1, "M2": M2,
        "lr": lr, "epochs": epochs,
        "ila": True,
    }

    return x_dpd_np, model