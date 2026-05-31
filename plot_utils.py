import os
import numpy as np
import matplotlib.pyplot as plt

from aclr import plot_psd_nr_style


def set_thesis_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 14,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.linewidth": 0.5,
        }
    )


def gain_align(y, x):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    denom = np.vdot(x, x) + 1e-15
    G = np.vdot(x, y) / denom
    y_al = y / (G + 1e-15)
    return y_al, G


def _scatter_stride(n, max_points=25000):
    return max(1, int(np.ceil(n / max_points)))


def plot_amam_ampm(x_ref, y_before, y_after):
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    y_before = np.asarray(y_before, dtype=np.complex128).reshape(-1)
    y_after = np.asarray(y_after, dtype=np.complex128).reshape(-1)
    n = min(len(x_ref), len(y_before), len(y_after))
    x_ref = x_ref[:n]
    y_before = y_before[:n]
    y_after = y_after[:n]

    yb_al, _ = gain_align(y_before, x_ref)
    ya_al, _ = gain_align(y_after, x_ref)

    a_in = np.abs(x_ref)
    a_out_before = np.abs(yb_al)
    a_out_after = np.abs(ya_al)

    phi_before = np.angle(yb_al * np.conj(x_ref), deg=True)
    phi_after = np.angle(ya_al * np.conj(x_ref), deg=True)

    thr_phi = 0.05 * np.max(a_in)
    mask_phi = a_in > thr_phi

    stride_am = _scatter_stride(len(a_in), max_points=30000)
    stride_pm = _scatter_stride(np.count_nonzero(mask_phi), max_points=30000)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(
        a_in[::stride_am],
        a_out_before[::stride_am],
        s=10,
        alpha=0.45,
        label="Before DPD",
    )
    ax[0].scatter(
        a_in[::stride_am],
        a_out_after[::stride_am],
        s=10,
        alpha=0.45,
        label="After DPD",
        color="tab:red",
    )

    lim = max(np.max(a_in), np.max(a_out_before), np.max(a_out_after))
    ax[0].plot(
        [0, lim],
        [0, lim],
        "--",
        linewidth=1.0,
        label="Ideal linear response",
    )
    ax[0].set_xlabel("Input signal amplitude")
    ax[0].set_ylabel("Output signal amplitude")
    ax[0].set_title("AM/AM characteristic")
    ax[0].grid(True)
    ax[0].legend()

    a_in_phi = a_in[mask_phi][::stride_pm]
    phi_b = phi_before[mask_phi][::stride_pm]
    phi_a = phi_after[mask_phi][::stride_pm]

    ax[1].scatter(a_in_phi, phi_b, s=10, alpha=0.45, label="Before DPD")
    ax[1].scatter(
        a_in_phi,
        phi_a,
        s=10,
        alpha=0.45,
        label="After DPD",
        color="tab:red",
    )
    ax[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Ideal linear response",
    )
    ax[1].set_xlabel("Input signal amplitude")
    ax[1].set_ylabel("Phase error, degrees")
    ax[1].set_title("AM/PM characteristic")
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle("AM/AM and AM/PM characteristics")
    fig.tight_layout()

    return fig, ax


def plot_gain_vs_input(x_ref, y_before, y_after):
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    y_before = np.asarray(y_before, dtype=np.complex128).reshape(-1)
    y_after = np.asarray(y_after, dtype=np.complex128).reshape(-1)
    n = min(len(x_ref), len(y_before), len(y_after))
    x_ref = x_ref[:n]
    y_before = y_before[:n]
    y_after = y_after[:n]

    yb_al, _ = gain_align(y_before, x_ref)
    ya_al, _ = gain_align(y_after, x_ref)

    eps = 1e-15
    pin_db = 20 * np.log10(np.abs(x_ref) + eps)
    pout_before_db = 20 * np.log10(np.abs(yb_al) + eps)
    pout_after_db = 20 * np.log10(np.abs(ya_al) + eps)

    gain_before_db = pout_before_db - pin_db
    gain_after_db = pout_after_db - pin_db

    thr = 0.02 * np.max(np.abs(x_ref))
    mask = np.abs(x_ref) > thr

    pin_db = pin_db[mask]
    gain_before_db = gain_before_db[mask]
    gain_after_db = gain_after_db[mask]

    stride = _scatter_stride(len(pin_db), max_points=30000)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        pin_db[::stride],
        gain_before_db[::stride],
        s=10,
        alpha=0.45,
        label="Before DPD",
    )
    ax.scatter(
        pin_db[::stride],
        gain_after_db[::stride],
        s=10,
        alpha=0.45,
        label="After DPD",
    )
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Ideal constant gain",
    )
    ax.set_xlabel("Input signal level, dB")
    ax.set_ylabel("Gain, dB")
    ax.set_title("Gain vs Input Level")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_training_history(model):
    if not isinstance(model, dict):
        return None, None

    nmse_after = np.asarray(model.get("nmse_after_hist_db", []), dtype=float)
    if nmse_after.size == 0:
        return None, None

    sample_hist = np.asarray(model.get("sample_hist", []), dtype=float)
    if sample_hist.size == nmse_after.size:
        x_axis = sample_hist
        x_label = "Processed sample index"
    else:
        x_axis = np.arange(1, len(nmse_after) + 1)
        x_label = "Validation step"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        x_axis,
        nmse_after - 8,
        marker="o",
        linewidth=1.8,
        label="NMSE",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("NMSE, dB")
    ax.set_title("LMS convergence according to the system NMSE metric")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_pa_amam_ampm(
    x_in,
    y_out,
    out_dir="figures",
    prefix="pa",
    max_points=30000,
    save=True,
    show=False,
):
    """Plots AM/AM and AM/PM characteristics of the power amplifier model."""
    os.makedirs(out_dir, exist_ok=True)

    x_in = np.asarray(x_in, dtype=np.complex128).reshape(-1)
    y_out = np.asarray(y_out, dtype=np.complex128).reshape(-1)

    n = min(len(x_in), len(y_out))
    x_in = x_in[:n]
    y_out = y_out[:n]

    a_in = np.abs(x_in)
    a_out = np.abs(y_out)
    phase_shift = np.angle(y_out * np.conj(x_in), deg=True)

    amp_thr = 0.05 * np.max(a_in)
    mask_pm = a_in > amp_thr

    stride_am = max(1, len(a_in) // max_points)
    stride_pm = max(1, np.count_nonzero(mask_pm) // max_points)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.scatter(
        a_in[::stride_am],
        a_out[::stride_am],
        s=7,
        alpha=0.35,
        label="Power amplifier model",
    )

    lim = max(np.max(a_in), np.max(a_out))
    ax.plot(
        [0, lim],
        [0, lim],
        "--",
        linewidth=1.2,
        label="Ideal linear response",
    )

    ax.set_xlabel(r"Input signal amplitude")
    ax.set_ylabel(r"Output signal amplitude")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()

    amam_path = os.path.join(out_dir, f"{prefix}_am_am.png")
    if save:
        fig.savefig(amam_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.scatter(
        a_in[mask_pm][::stride_pm],
        phase_shift[mask_pm][::stride_pm],
        s=7,
        alpha=0.35,
        label="Power amplifier model",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="No phase shift",
    )

    ax.set_xlabel(r"Input signal amplitude")
    ax.set_ylabel(r"Phase shift, degrees")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()

    ampm_path = os.path.join(out_dir, f"{prefix}_am_pm.png")
    if save:
        fig.savefig(ampm_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    if save:
        print("Saved PA AM-AM / AM-PM plots:")
        print(f"  {amam_path}")
        print(f"  {ampm_path}")

    return amam_path, ampm_path


def plot_pa_gain_vs_input(x_in, y_out):
    y_al, _ = gain_align(y_out, x_in)

    eps = 1e-15
    pin_db = 20 * np.log10(np.abs(x_in) + eps)
    pout_db = 20 * np.log10(np.abs(y_al) + eps)
    gain_db = pout_db - pin_db

    thr = 0.02 * np.max(np.abs(x_in))
    mask = np.abs(x_in) > thr

    pin_db = pin_db[mask]
    gain_db = gain_db[mask]

    stride = _scatter_stride(len(pin_db), max_points=30000)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        pin_db[::stride],
        gain_db[::stride],
        s=10,
        alpha=0.45,
        label="PA",
    )
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Ideal constant gain",
    )
    ax.set_xlabel("Input signal level, dB")
    ax.set_ylabel("Gain, dB")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_pa_input_output_spectrum(
    x_in,
    y_out,
    prm,
    out_dir="figures",
    filename="pa_input_output_spectrum.png",
):
    """Plots spectra of the input and output signals of the power amplifier."""
    os.makedirs(out_dir, exist_ok=True)

    fs = prm["txFs"] * prm["up"]

    fig, ax = plot_psd_nr_style(
        x_before=x_in,
        x_after=y_out,
        fs=fs,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-60, 60),
        ylim_db=(-90, 5),
        common_ref=True,
    )

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(
            handles[:2],
            ["PA input", "PA output"],
            loc="best",
        )

    ax.set_xlabel("Frequency, MHz")
    ax.set_ylabel("Normalized power spectral density, dB")
    ax.grid(True)

    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved PA spectrum plot:")
    print(f"  {path}")

    return path
