import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

from generator import generator
from amp_model import amp_model
from aclr import plot_psd_nr_style, plot_aclr_nr_style

from ls_alg import (
    align_by_xcorr,
    ls_postdistorter_coeffs,
    apply_predistorter,
    nmse_db_gain_aligned,
)
from lms_alg import (
    train_lms_feedback_predistorter,
    apply_mp_feedback_predistorter,
    estimate_integer_lag,
    align_pair_by_lag,
)
from cnn_dpd import cnn_dpd
from cnn_dpd_torch import (
    cnn_dpd_torch,
    CNNPostDistorter,
    apply_predistorter_torch,
)
from plot_utils import (
    set_thesis_plot_style,
    gain_align,
    plot_amam_ampm,
    plot_gain_vs_input,
    plot_training_history,
    plot_pa_amam_ampm,
    plot_pa_gain_vs_input,
    plot_pa_input_output_spectrum,
)


def build_params():
    prm = {
        "sizeSig": int(4e4),
        "txFs": 30.72e6,
        "sigBand": 20e6,
        "up": 8,
        "signal_mode": "ofdm",  # 'noise' or 'ofdm'
        # generator seeds
        "signal_seed_train": 101,
        "signal_seed_test": 202,
        "plot_signal": False,
        "plot_dir": "figures",
        "plot_prefix": "ofdm_input_signal",
        "iq_plot_samples": 2000,
        # PA params
        "pa_mode": "gmp",
        "pa_alpha": 0.8,
        "mem_decay": 0.7,
        "pa_memory": 3,
        "gmp_beta": 0.15,
        "gmp_lead_beta": 0.05,
        "gmp_aligned_orders": [1, 3, 5],
        "gmp_aligned_memory": 3,
        "gmp_lag_orders": [1, 3, 5],
        "gmp_lag_memory": 3,
        "gmp_lag_env_delays": [],  # [1, 2],
        # Для отладки LMS лучше временно отключить lead terms,
        # потому что direct online LMS чувствителен к некаузальным членам GMP.
        # Для финальной GMP-модели можно вернуть [1, 3, 5] и [1, 2].
        "gmp_lead_orders": [1, 3, 5],
        "gmp_lead_env_delays": [1, 2],
        # "gmp_lead_orders": [],
        # "gmp_lead_env_delays": [],
        "gmp_lead_memory": 3,
        # OFDM-like params
        "ofdm_nfft": 1024,
        "ofdm_scs": 30e3,
        "ofdm_n_rb": 51,
        "ofdm_cp_len": 72,
        "ofdm_qam_order": 1024,
        "ofdm_passband": 9e6,
        "ofdm_stopband": 10.5e6,
        "ofdm_filter_atten_db": 70.0,
        "ofdm_post_num_taps": 201,
    }

    prm["cnn"] = {
        "memory": 5,
        "kernel": 5,
        "filters": 6,
        "M1": 8,
        "epochs": 70,
        "lr": 1e-3,
        "seed": 42,
        "features": "poly",
        "print_every": 5,
        "clip": 1.0,
        "weight_decay": 0.0,
        "ila_iters": 10,
        "warm_start": True,
        "batch_size": 4096,
        "batch_mode": "contig",
        "residual": True,
        "power_constraint": True,
        "device": "cpu",
    }

    prm["lms"] = {
        "orders": (1, 3, 5),
        "memory_depth": 3,
        "mu": 1e-4,
        "epochs": 1,
        "block_size": 1,
        "context_len": 64,
        "right_context": 64,
        "feedback_gain": 0.3,
        "update_sign": 1.0,
        "normalized": False,
        "delay": 0,
        "max_lag": 20,
        "use_gain": True,
        "use_block_gain": False,
        "gain_ref": "x",
        "power_constraint": True,
        "coef_leak": 0.0,
        "max_coef_norm": None,
        "min_amp_ratio": 0.03,
        "print_every": 20000,
        "eval_every": 20000,
        "max_eval_len": 20000,
        "keep_best": True,
    }

    return prm


def run_dpd(method, cnn_backend, x_al, y_al, prm):
    method = method.lower()

    if method == "ls":
        orders = (1, 3, 5)
        memory_depth = 8
        ridge = 1e-2

        a = ls_postdistorter_coeffs(
            y_al,
            x_al,
            orders=orders,
            memory_depth=memory_depth,
            ridge=ridge,
            normalize_gain=True,
        )

        x_dpd = apply_predistorter(
            x_al,
            a,
            orders=orders,
            memory_depth=memory_depth,
        )

        model = {
            "a": a,
            "orders": orders,
            "memory_depth": memory_depth,
            "ridge": ridge,
            "kind": "mp_ls",
        }

    elif method == "lms":
        x_dpd, model = train_lms_feedback_predistorter(
            x_al=x_al,
            y_al=y_al,
            pa_fn=lambda z: amp_model(prm, z),
            lms_prm=prm["lms"],
        )

    elif method == "cnn":
        if cnn_backend == "torch":
            x_dpd, model = cnn_dpd_torch(
                x_al,
                y_al,
                prm,
                pa_fn=lambda z: amp_model(prm, z),
            )
            model["kind"] = "cnn_torch"
        elif cnn_backend == "numpy":
            x_dpd, model = cnn_dpd(x_al, y_al, prm)
            model["kind"] = "cnn_numpy"
        else:
            raise ValueError('cnn_backend must be "torch" or "numpy"')

    else:
        raise ValueError('method must be "ls" or "lms" or "cnn"')

    return x_dpd, model


def generate_aligned_pair(prm, signal_seed):
    prm_sig = copy.deepcopy(prm)
    prm_sig["seed"] = int(signal_seed)

    sig = generator(prm_sig)
    x = sig / (np.max(np.abs(sig)) + 1e-15)
    y = amp_model(prm_sig, x)

    x_al, y_al, lag = align_by_xcorr(x, y, max_lag=300)
    return x_al, y_al, lag


def normalize_drive(x_ref, x_dpd):
    p_ref = np.mean(np.abs(x_ref) ** 2) + 1e-15
    p_dpd = np.mean(np.abs(x_dpd) ** 2) + 1e-15
    x_dpd_n = x_dpd * np.sqrt(p_ref / p_dpd)
    return x_dpd_n, p_ref, p_dpd


def apply_saved_cnn_torch(model_dict, x_ref):
    features = model_dict["features"]
    K = int(model_dict["K"])
    Ff = int(model_dict["F"])
    M1 = int(model_dict["M1"])
    residual = bool(model_dict["residual"])
    power_constraint = bool(model_dict.get("power_constraint", True))
    feat_rms_np = np.asarray(model_dict["feat_rms"], dtype=np.float32)
    state_dict = model_dict["torch_state_dict"]
    device = torch.device(model_dict.get("device", "cpu"))

    C = 6 if features.lower() == "poly" else 2

    model = CNNPostDistorter(
        C=C,
        K=K,
        Ff=Ff,
        M1=M1,
        residual=residual,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x_t = torch.from_numpy(np.asarray(x_ref, dtype=np.complex128)).to(device)
    feat_rms_t = torch.from_numpy(feat_rms_np).to(device)

    with torch.no_grad():
        x_dpd_t = apply_predistorter_torch(
            model=model,
            x=x_t,
            features=features,
            feat_rms=feat_rms_t,
            device=device,
            power_constraint=power_constraint,
        )

    return x_dpd_t.detach().cpu().numpy().astype(np.complex128)


def apply_model_on_signal(method, cnn_backend, model, x_ref):
    method = method.lower()

    if method == "ls":
        return apply_predistorter(
            x_ref,
            model["a"],
            orders=model["orders"],
            memory_depth=model["memory_depth"],
        )

    if method == "lms":
        return apply_mp_feedback_predistorter(
            x_ref,
            model["a"],
            orders=model["orders"],
            memory_depth=model["memory_depth"],
        )

    if method == "cnn":
        if cnn_backend == "torch":
            return apply_saved_cnn_torch(model, x_ref)
        raise NotImplementedError(
            "Test-time apply for numpy CNN is not implemented here."
        )

    raise ValueError('method must be "ls" or "lms" or "cnn"')


def style_before_after_lines(ax, before_label="До DPD", after_label="После DPD"):
    lines = ax.get_lines()

    if len(lines) >= 1:
        lines[0].set_color("tab:blue")
        lines[0].set_linewidth(2.4)
        lines[0].set_label(before_label)

    if len(lines) >= 2:
        lines[1].set_color("tab:red")
        lines[1].set_linewidth(2.4)
        lines[1].set_label(after_label)


def evaluate_case(tag, prm, method, cnn_backend, model, x_ref, y_ref, make_plots=False):
    x_dpd = apply_model_on_signal(method, cnn_backend, model, x_ref)
    x_dpd, p_ref, p_dpd = normalize_drive(x_ref, x_dpd)

    # y_ref уже выровнен в generate_aligned_pair().
    # y_lin_raw — сырой выход PA после применения DPD, его надо выровнять заново.
    y_lin_raw = amp_model(prm, x_dpd)

    lag_after = estimate_integer_lag(
        x_ref,
        y_lin_raw,
        max_lag=300,
    )

    x_eval, y_lin = align_pair_by_lag(
        x_ref,
        y_lin_raw,
        lag_after,
    )

    y_ref_eval = y_ref[: len(x_eval)]

    nmse_before = nmse_db_gain_aligned(y_ref_eval, x_eval)
    nmse_after = nmse_db_gain_aligned(y_lin, x_eval)

    print(f"\n[{tag}]")
    print(f"Output alignment lag after DPD: {lag_after} samples")
    print(f"Input RMS power before DPD drive norm: {10*np.log10(p_dpd):.2f} dB")
    print(f"Reference RMS power: {10*np.log10(p_ref):.2f} dB")
    print(f"Gain-aligned NMSE before DPD: {nmse_before:.2f} dB")
    print(f"Gain-aligned NMSE after  DPD: {nmse_after:.2f} dB")

    fs = prm["txFs"] * prm["up"]
    bw_aclr = prm["sigBand"]

    plot_dir = prm.get("plot_dir", "figures")
    os.makedirs(plot_dir, exist_ok=True)

    # -------------------------------------------------
    # PSD before/after DPD
    # -------------------------------------------------
    fig0, ax0 = plot_psd_nr_style(
        x_before=y_ref_eval,
        x_after=y_lin,
        fs=fs,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-50, 50),
        ylim_db=(-60, 0),
        title="Спектральная плотность мощности на выходе усилителя",
        common_ref=True,
    )

    style_before_after_lines(
        ax0,
        before_label="До DPD",
        after_label="После DPD",
    )

    ax0.set_xlim(-50, 50)
    ax0.set_ylim(-60, 0)
    ax0.set_xlabel("Частота, МГц")
    ax0.set_ylabel("Спектральная плотность мощности, дБ")
    ax0.set_title("Спектральная плотность мощности на выходе усилителя")
    ax0.grid(True)
    ax0.legend(loc="upper right")

    fig0.tight_layout()

    psd_path = os.path.join(
        plot_dir,
        f"{tag.lower()}_pa_output_psd_before_after_dpd.png",
    )
    fig0.savefig(psd_path, dpi=300, bbox_inches="tight")
    print(f"[{tag}] Saved PSD plot: {psd_path}")

    # -------------------------------------------------
    # ACLR before/after DPD
    # -------------------------------------------------
    fig, ax, met = plot_aclr_nr_style(
        x_before=y_ref_eval,
        x_after=y_lin,
        fs=fs,
        bw=bw_aclr,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-50, 50),
        ylim_db=(-60, 0),
        title=f"ACLR для сигнала на выходе усилителя [{tag}]",
        common_ref=True,
    )

    style_before_after_lines(
        ax,
        before_label="До DPD",
        after_label="После DPD",
    )

    ax.set_xlim(-50, 50)
    ax.set_ylim(-60, 0)
    ax.set_xlabel("Частота, МГц")
    ax.set_ylabel("Спектральная плотность мощности, дБ")
    ax.set_title(f"ACLR для сигнала на выходе усилителя [{tag}]")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()

    aclr_path = os.path.join(
        plot_dir,
        f"{tag.lower()}_pa_output_aclr_before_after_dpd.png",
    )
    fig.savefig(aclr_path, dpi=300, bbox_inches="tight")
    print(f"[{tag}] Saved ACLR plot: {aclr_path}")

    print(f"[{tag}] До DPD:")
    print(f"  ACLR(-1) = {met['before']['aclr_m1_db']:.2f} dB")
    print(f"  ACLR(+1) = {met['before']['aclr_p1_db']:.2f} dB")
    print(f"  Leakage(-1) = {met['before']['leak_m1_dbc']:.2f} dBc")
    print(f"  Leakage(+1) = {met['before']['leak_p1_dbc']:.2f} dBc")

    print(f"[{tag}] После DPD:")
    print(f"  ACLR(-1) = {met['after']['aclr_m1_db']:.2f} dB")
    print(f"  ACLR(+1) = {met['after']['aclr_p1_db']:.2f} dB")
    print(f"  Leakage(-1) = {met['after']['leak_m1_dbc']:.2f} dBc")
    print(f"  Leakage(+1) = {met['after']['leak_p1_dbc']:.2f} dBc")

    if make_plots:
        plot_amam_ampm(x_eval, y_ref_eval, y_lin)
        plot_gain_vs_input(x_eval, y_ref_eval, y_lin)

    return {
        "x_dpd": x_dpd,
        "y_lin": y_lin,
        "lag_after": lag_after,
        "nmse_before_db": nmse_before,
        "nmse_after_db": nmse_after,
        "aclr": met,
        "psd_path": psd_path,
        "aclr_path": aclr_path,
    }


def main():
    plt.close("all")
    plt.rcParams["font.family"] = "DejaVu Sans"

    method = "cnn"  # "ls", "lms", "cnn"
    cnn_backend = "torch"  # "torch" or "numpy"

    prm = build_params()

    # -----------------------------
    # Train waveform
    # -----------------------------
    x_train, y_train, lag_train = generate_aligned_pair(prm, prm["signal_seed_train"])
    print(
        f"[TRAIN] Alignment lag = {lag_train} samples. Using aligned length = {len(x_train)}"
    )

    set_thesis_plot_style()

    eps = 1e-12
    norm = np.max(np.abs(x_train)) + eps

    x_plot = x_train / norm
    y_plot = y_train / norm

    alpha = np.vdot(y_plot, x_plot) / (np.vdot(y_plot, y_plot) + eps)
    y_plot_aligned = alpha * y_plot

    plot_pa_input_output_spectrum(
        x_plot,
        y_plot_aligned,
        prm,
        out_dir="figures",
        filename="pa_input_output_spectrum.png",
    )

    plot_pa_amam_ampm(
        x_plot,
        y_plot_aligned,
        out_dir="figures",
        prefix="pa",
        max_points=30000,
        save=True,
        show=False,
    )

    # Train DPD on train waveform
    _, model = run_dpd(method, cnn_backend, x_train, y_train, prm)

    plot_pa_gain_vs_input(x_train, y_train)

    # -----------------------------
    # Evaluate on the same train waveform
    # -----------------------------
    train_res = evaluate_case(
        tag="TRAIN",
        prm=prm,
        method=method,
        cnn_backend=cnn_backend,
        model=model,
        x_ref=x_train,
        y_ref=y_train,
        make_plots=False,
    )

    # -----------------------------
    # Independent test waveform
    # -----------------------------
    x_test, y_test, lag_test = generate_aligned_pair(prm, prm["signal_seed_test"])
    print(
        f"\n[TEST] Alignment lag = {lag_test} samples. Using aligned length = {len(x_test)}"
    )

    test_res = evaluate_case(
        tag="TEST",
        prm=prm,
        method=method,
        cnn_backend=cnn_backend,
        model=model,
        x_ref=x_test,
        y_ref=y_test,
        make_plots=True,
    )

    plot_training_history(model)

    plt.show()


if __name__ == "__main__":
    main()
