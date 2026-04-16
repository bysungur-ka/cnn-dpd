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
    nmse_db_gain_aligned
)
from lms_alg import lms_postdistorter_coeffs

from cnn_dpd import cnn_dpd
from cnn_dpd_torch import (
    cnn_dpd_torch,
    CNNPostDistorter,
    apply_predistorter_torch,
)


def build_params():
    prm = {
        'sizeSig': int(4e4),
        'txFs': 30.72e6,
        'sigBand': 20e6,
        'up': 8,
        'signal_mode': 'noise',   # 'noise' or 'ofdm'

        # generator seeds
        'signal_seed_train': 101,
        'signal_seed_test': 202,

        # PA params
        'pa_mode': 'gmp',
        'pa_alpha': 0.8,
        'pa_memory': 3,
        'mem_decay': 0.7,
        'gmp_beta': 0.15,

        'gmp_aligned_orders': [1, 3, 5],
        'gmp_aligned_memory': 3,

        'gmp_lag_orders': [3],
        'gmp_lag_memory': 3,
        'gmp_lag_env_delays': [2],

        # сначала лучше выключить leading:
        'gmp_lead_orders': [],
        'gmp_lead_memory': 3,
        'gmp_lead_env_delays': [],
        # MP model params
        'mp_orders': [1, 3, 5],
        'mp_memory_depth': 3,
        'mp_alpha': 0.8,

        # IIR mode params
        'pa_b': [0.85, 0.12],
        'pa_a': [1.0, -0.55, 0.16],
        'pa_gain': 1.0,

        # OFDM-like params
        'ofdm_nfft': 1024,
        'ofdm_scs': 30e3,
        'ofdm_n_rb': 51,
        'ofdm_cp_len': 72,
        'ofdm_qam_order': 1024,
        'ofdm_passband': 9e6,
        'ofdm_stopband': 10.5e6,
        'ofdm_filter_atten_db': 70.0,
        'ofdm_post_num_taps': 201,
    }

    prm['cnn'] = {
        'memory': 5,
        'kernel': 5,
        'filters': 6,
        'M1': 8,
        'epochs': 120,
        'lr': 1e-3,
        'seed': 42,
        'features': 'poly',
        'print_every': 10,
        'clip': 1.0,
        'weight_decay': 0.0,
        'ila_iters': 10,
        'warm_start': True,
        'batch_size': 4096,
        'batch_mode': 'contig',
        'residual': True,
        'power_constraint': True,
        'device': 'cpu',
    }

    return prm


def run_dpd(method, cnn_backend, x_al, y_al, prm):
    if method.lower() == "ls":
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
            'a': a,
            'orders': orders,
            'memory_depth': memory_depth,
            'ridge': ridge,
            'kind': 'mp_ls',
        }

    elif method.lower() == "lms":
        orders = (1, 3, 5)
        mu = 1e-2
        epochs_lms = 100

        a = lms_postdistorter_coeffs(
            y_al,
            x_al,
            orders=orders,
            mu=mu,
            epochs=epochs_lms
        )

        x_dpd = apply_predistorter(x_al, a, orders=orders)
        model = {
            'a': a,
            'orders': orders,
            'mu': mu,
            'epochs': epochs_lms,
            'kind': 'lms',
        }

    elif method.lower() == "cnn":
        if cnn_backend == "torch":
            x_dpd, model = cnn_dpd_torch(
                x_al,
                y_al,
                prm,
                pa_fn=lambda z: amp_model(prm, z)
            )
            model['kind'] = 'cnn_torch'
        elif cnn_backend == "numpy":
            x_dpd, model = cnn_dpd(x_al, y_al, prm)
            model['kind'] = 'cnn_numpy'
        else:
            raise ValueError('cnn_backend must be "torch" or "numpy"')

    else:
        raise ValueError('method must be "ls" or "lms" or "cnn"')

    return x_dpd, model


def gain_align(y, x):
    denom = np.vdot(x, x) + 1e-15
    G = np.vdot(x, y) / denom
    y_al = y / (G + 1e-15)
    return y_al, G


def _scatter_stride(n, max_points=25000):
    return max(1, int(np.ceil(n / max_points)))


def plot_amam_ampm(x_ref, y_before, y_after):
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
        label='До DPD'
    )
    ax[0].scatter(
        a_in[::stride_am],
        a_out_after[::stride_am],
        s=10,
        alpha=0.45,
        label='После DPD'
    )

    lim = max(np.max(a_in), np.max(a_out_before), np.max(a_out_after))
    ax[0].plot([0, lim], [0, lim], '--', linewidth=1.0, label='Идеальная линейность')
    ax[0].set_xlabel('Амплитуда входного сигнала')
    ax[0].set_ylabel('Амплитуда выходного сигнала')
    ax[0].set_title('AM/AM характеристика')
    ax[0].grid(True)
    ax[0].legend()

    a_in_phi = a_in[mask_phi][::stride_pm]
    phi_b = phi_before[mask_phi][::stride_pm]
    phi_a = phi_after[mask_phi][::stride_pm]

    ax[1].scatter(
        a_in_phi,
        phi_b,
        s=10,
        alpha=0.45,
        label='До DPD'
    )
    ax[1].scatter(
        a_in_phi,
        phi_a,
        s=10,
        alpha=0.45,
        label='После DPD'
    )
    ax[1].axhline(0.0, linestyle='--', linewidth=1.0, label='Идеальная линейность')
    ax[1].set_xlabel('Амплитуда входного сигнала')
    ax[1].set_ylabel('Фазовая ошибка, градусы')
    ax[1].set_title('AM/PM характеристика')
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle('Амплитудно-амплитудная и амплитудно-фазовая характеристики каскада')
    fig.tight_layout()


def plot_gain_vs_input(x_ref, y_before, y_after):
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

    plt.figure(figsize=(6, 5))
    plt.scatter(
        pin_db[::stride],
        gain_before_db[::stride],
        s=10,
        alpha=0.45,
        label='До DPD'
    )
    plt.scatter(
        pin_db[::stride],
        gain_after_db[::stride],
        s=10,
        alpha=0.45,
        label='После DPD'
    )
    plt.axhline(0.0, linestyle='--', linewidth=1.0, label='Идеально постоянное усиление')
    plt.xlabel('Уровень входного сигнала, дБ')
    plt.ylabel('Коэффициент усиления, дБ')
    plt.title('Gain vs Input Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_ila_history(model):
    if not isinstance(model, dict):
        return

    nmse_after = np.asarray(model.get('nmse_after_hist_db', []), dtype=float)
    if nmse_after.size == 0:
        return

    plt.figure()
    plt.plot(
        np.arange(1, len(nmse_after) + 1),
        nmse_after,
        marker='o',
        linewidth=1.8,
        label='NMSE'
    )
    plt.xlabel('Номер итерации ILA')
    plt.ylabel('NMSE, дБ')
    plt.title('Сходимость ILA по системной метрике NMSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def generate_aligned_pair(prm, signal_seed):
    prm_sig = copy.deepcopy(prm)
    prm_sig['seed'] = int(signal_seed)

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
    features = model_dict['features']
    K = int(model_dict['K'])
    Ff = int(model_dict['F'])
    M1 = int(model_dict['M1'])
    residual = bool(model_dict['residual'])
    power_constraint = bool(model_dict.get('power_constraint', True))
    feat_rms_np = np.asarray(model_dict['feat_rms'], dtype=np.float32)
    state_dict = model_dict['torch_state_dict']
    device = torch.device(model_dict.get('device', 'cpu'))

    C = 6 if features.lower() == 'poly' else 2

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
    if method.lower() == "ls":
        return apply_predistorter(
            x_ref,
            model['a'],
            orders=model['orders'],
            memory_depth=model['memory_depth'],
        )

    if method.lower() == "lms":
        return apply_predistorter(
            x_ref,
            model['a'],
            orders=model['orders'],
        )

    if method.lower() == "cnn":
        if cnn_backend == "torch":
            return apply_saved_cnn_torch(model, x_ref)
        raise NotImplementedError("Test-time apply for numpy CNN is not implemented here.")

    raise ValueError('method must be "ls" or "lms" or "cnn"')


def evaluate_case(tag, prm, method, cnn_backend, model, x_ref, y_ref, make_plots=False):
    x_dpd = apply_model_on_signal(method, cnn_backend, model, x_ref)
    x_dpd, p_ref, p_dpd = normalize_drive(x_ref, x_dpd)

    y_lin = amp_model(prm, x_dpd)

    nmse_before = nmse_db_gain_aligned(y_ref, x_ref)
    nmse_after = nmse_db_gain_aligned(y_lin, x_ref)

    print(f"\n[{tag}]")
    print(f"Input RMS power before DPD drive norm: {10*np.log10(p_dpd):.2f} dB")
    print(f"Reference RMS power: {10*np.log10(p_ref):.2f} dB")
    print(f"Gain-aligned NMSE before DPD: {nmse_before:.2f} dB")
    print(f"Gain-aligned NMSE after  DPD: {nmse_after:.2f} dB")

    fs = prm['txFs'] * prm['up']
    bw_aclr = prm['sigBand'] + 5e6

    fig0, ax0 = plot_psd_nr_style(
        x_before=y_ref,
        x_after=y_lin,
        fs=fs,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-100, 100),
        ylim_db=(-60, 5),
        title=f'Спектральная плотность мощности на выходе усилителя [{tag}]',
        common_ref=True
    )

    fig, ax, met = plot_aclr_nr_style(
        x_before=y_ref,
        x_after=y_lin,
        fs=fs,
        bw=bw_aclr,
        nperseg=4096,
        noverlap=2048,
        xlim_mhz=(-100, 100),
        ylim_db=(-60, 5),
        title=f'ACLR для сигнала на выходе усилителя [{tag}]',
        common_ref=True
    )

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
        plot_amam_ampm(x_ref, y_ref, y_lin)
        plot_gain_vs_input(x_ref, y_ref, y_lin)

    return {
        'x_dpd': x_dpd,
        'y_lin': y_lin,
        'nmse_before_db': nmse_before,
        'nmse_after_db': nmse_after,
        'aclr': met,
    }

def plot_pa_amam_ampm(x_in, y_out):
    y_al, _ = gain_align(y_out, x_in)

    a_in = np.abs(x_in)
    a_out = np.abs(y_al)
    phi = np.angle(y_al * np.conj(x_in), deg=True)

    thr_phi = 0.05 * np.max(a_in)
    mask_phi = a_in > thr_phi

    stride_am = _scatter_stride(len(a_in), max_points=30000)
    stride_pm = _scatter_stride(np.count_nonzero(mask_phi), max_points=30000)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # AM/AM
    ax[0].scatter(
        a_in[::stride_am],
        a_out[::stride_am],
        s=10,
        alpha=0.45,
        label='PA'
    )

    lim = max(np.max(a_in), np.max(a_out))
    ax[0].plot([0, lim], [0, lim], '--', linewidth=1.0, label='Идеальная линейность')

    ax[0].set_xlabel('Амплитуда входного сигнала')
    ax[0].set_ylabel('Амплитуда выходного сигнала')
    ax[0].set_title('AM/AM характеристика усилителя')
    ax[0].grid(True)
    ax[0].legend()

    # AM/PM
    ax[1].scatter(
        a_in[mask_phi][::stride_pm],
        phi[mask_phi][::stride_pm],
        s=10,
        alpha=0.45,
        label='PA'
    )
    ax[1].axhline(0.0, linestyle='--', linewidth=1.0, label='Идеальная линейность')

    ax[1].set_xlabel('Амплитуда входного сигнала')
    ax[1].set_ylabel('Фазовая ошибка, градусы')
    ax[1].set_title('AM/PM характеристика усилителя')
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle('Характеристики усилителя мощности')
    fig.tight_layout()
    
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

    plt.figure(figsize=(6, 5))
    plt.scatter(
        pin_db[::stride],
        gain_db[::stride],
        s=10,
        alpha=0.45,
        label='PA'
    )
    plt.axhline(0.0, linestyle='--', linewidth=1.0, label='Идеально постоянное усиление')
    plt.xlabel('Уровень входного сигнала, дБ')
    plt.ylabel('Коэффициент усиления, дБ')
    plt.title('Gain vs Input Level для усилителя')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    

def main():
    plt.close('all')
    plt.rcParams['font.family'] = 'DejaVu Sans'

    method = "ls"        # "ls", "lms", "cnn"
    cnn_backend = "torch" # "torch" or "numpy"

    prm = build_params()

    # -----------------------------
    # Train waveform
    # -----------------------------
    x_train, y_train, lag_train = generate_aligned_pair(prm, prm['signal_seed_train'])
    print(f"[TRAIN] Alignment lag = {lag_train} samples. Using aligned length = {len(x_train)}")

    # Train DPD on train waveform
    _, model = run_dpd(method, cnn_backend, x_train, y_train, prm)
    
    plot_pa_amam_ampm(x_train, y_train)
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
    x_test, y_test, lag_test = generate_aligned_pair(prm, prm['signal_seed_test'])
    print(f"\n[TEST] Alignment lag = {lag_test} samples. Using aligned length = {len(x_test)}")

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

    # CNN ILA history makes sense to show once
    plot_ila_history(model)

    plt.show()


if __name__ == "__main__":
    main()