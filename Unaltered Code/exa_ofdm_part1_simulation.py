"""
================================================================================
 ExA: Introduction to OFDM and Software Defined Radio
 PART 1 - SIMULATION-BASED OFDM  (Tasks 1 to 6)
================================================================================

Part 1 of the activity:

    Task 1  Signals and FFT review
    Task 2  Digital symbols (BPSK / QPSK) and constellations
    Task 3  Building a simple OFDM symbol (IFFT/FFT loop-back)
    Task 4  Adding noise (AWGN, SNR, BER vs SNR)
    Task 5  Cyclic prefix and simple multipath
    Task 6  Frequency offset (and optional correction)

How to run
----------
    python exa_ofdm_part1_simulation.py        # runs all tasks, shows figures
    SAVE_FIGS=1 python exa_ofdm_part1_simulation.py # also saves PNGs to ./figures/

Requirements:  numpy, scipy, matplotlib


Suggestions for exploring the simulation
----------------------------------------
This is a complete, working simulation. You do not need to add any code.
Your aim is to understand how the key parameters affect an OFDM link.
The suggestions below are starting points. Change a value inside the
relevant task function, re-run the script, and compare the figures and
printed results with those of the unchanged run. Try to explain what you see.

    Task 1  fs, N, f0 : move the tone, or sample below 2*f0 and find the alias.
            fa, fb    : bring the two tones closer; how large must N be to
                        still separate them (df = fs/N)?
    Task 3  active    : use more or different subcarriers, e.g. np.arange(1, 33).
    Task 4  [np.inf, 20, 5] : other SNRs for the constellation plots.
            n_symbols : more symbols -> lower BERs can be measured.
            active    : with more active subcarriers the curves shift right
                        (less SNR per subcarrier). Why?
    Task 5  cp_len    : make it shorter than the channel memory (e.g. 4 with
                        the delay-5 channel). What happens to BER and EVM?
            channels  : add your own echo, e.g. stronger or longer delay.
    Task 6  offsets   : try other fractions of the subcarrier spacing; at
                        which offset do QPSK errors start?
            correction: correct with a slightly wrong value (e.g. -0.9*foff
                        instead of -foff) and look at the residual rotation.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ------------------------------------------------------------------ #
#  Global configuration                           #
# ------------------------------------------------------------------ #
RNG = np.random.default_rng(seed=2024)      # reproducible random numbers
SAVE_FIGS = os.environ.get("SAVE_FIGS", "0") == "1"
FIG_DIR = "figures"
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)


def _finish(fig, name):
    """Save (optionally) and lay out a figure.  Call plt.show() once at the end."""
    fig.tight_layout()
    if SAVE_FIGS:
        path = os.path.join(FIG_DIR, name + ".png")
        fig.savefig(path, dpi=130)
        print(f"   [saved] {path}")


# ================================================================== #
#  Shared building blocks (used by several tasks)                    #
# ================================================================== #

# ---- Bit / symbol mapping -------------------------------------------------- #
def random_bits(n_bits):
    """Return an array of n_bits random 0/1 integers."""
    return RNG.integers(0, 2, size=n_bits)


def bpsk_modulate(bits):
    """BPSK:  bit 0 -> -1,  bit 1 -> +1.  Returns real-valued symbols."""
    return 2 * bits - 1            # 0->-1, 1->+1


def bpsk_demodulate(symbols):
    """BPSK decision: value > 0 -> bit 1, value < 0 -> bit 0."""
    return (np.real(symbols) > 0).astype(int)


# QPSK Gray mapping (matches the activity sheet):
#   00 -> ( 1 + j)/sqrt(2)
#   01 -> (-1 + j)/sqrt(2)
#   11 -> (-1 - j)/sqrt(2)
#   10 -> ( 1 - j)/sqrt(2)
# Equivalently, for a bit pair (b0, b1):
#   I  =  1 - 2*b1   (real part sign set by the 2nd bit)
#   Q  =  1 - 2*b0   (imag part sign set by the 1st bit)
# The 1/sqrt(2) normalises the average symbol power to 1.
_INV_SQRT2 = 1 / np.sqrt(2)


def qpsk_modulate(bits):
    """Map an even-length bit array to complex QPSK symbols (power-normalised)."""
    assert len(bits) % 2 == 0, "QPSK needs an even number of bits"
    b = bits.reshape(-1, 2)
    b0, b1 = b[:, 0], b[:, 1]
    i = 1 - 2 * b1
    q = 1 - 2 * b0
    return (i + 1j * q) * _INV_SQRT2


def qpsk_demodulate(symbols):
    """Nearest-quadrant decision for QPSK; returns the recovered bit array."""
    s = np.asarray(symbols)
    b1 = (np.real(s) < 0).astype(int)   # real<0 -> b1=1
    b0 = (np.imag(s) < 0).astype(int)   # imag<0 -> b0=1
    return np.column_stack((b0, b1)).reshape(-1)


# ---- OFDM transmitter / receiver ------------------------------------------- #
def ofdm_modulate(symbols, N, active):
    """
    Place modulation symbols on the chosen FFT bins and IFFT to time domain.

        symbols : modulation symbols, one per active subcarrier
        N       : FFT size (number of bins)
        active  : indices of the active subcarriers
    Returns the length-N time-domain OFDM symbol.
    """
    X = np.zeros(N, dtype=complex)   # frequency-domain array, all bins zero
    X[active] = symbols              # fill the active subcarriers
    return np.fft.ifft(X)            # IFFT -> time-domain waveform


def ofdm_demodulate(rx_time, N, active):
    """FFT the received block and extract the active subcarriers."""
    Y = np.fft.fft(rx_time)
    return Y[active]


# ---- AWGN ------------------------------------------------------------------ #
def add_awgn(signal, snr_db):
    """
    Add complex AWGN at the requested SNR (dB).  Noise power is referenced to
    the measured average power of `signal`, as in the activity sheet:
        noise_power = signal_power / 10**(snr_db/10)
    The complex noise is split equally between real and imaginary parts.
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        RNG.standard_normal(signal.shape) + 1j * RNG.standard_normal(signal.shape)
    )
    return signal + noise, noise_power


def ber(tx_bits, rx_bits):
    """Bit error rate between two equal-length bit arrays."""
    n = min(len(tx_bits), len(rx_bits))
    return np.sum(tx_bits[:n] != rx_bits[:n]) / n


# ================================================================== #
#  TASK 1 :  Signals and FFT review                                  #
# ================================================================== #
def task1_signals_and_fft():
    print("\n=== Task 1: Signals and FFT review ===")

    # --- (a) a single sinusoid ------------------------------------ #
    fs = 1000.0          # sampling rate [Hz]
    N = 512              # number of samples
    n = np.arange(N)
    t = n / fs           # time axis  t = n/fs
    A, f0 = 1.0, 50.0    # amplitude and frequency
    x = A * np.cos(2 * np.pi * f0 * t)

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1 / fs)          # bin centre frequencies
    mag = np.abs(X) / N                          # normalise magnitude

    # plot only the positive half of the spectrum (real signal -> symmetric)
    half = N // 2
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(t[:200], x[:200])
    ax[0].set(title=f"Single sinusoid  f = {f0:.0f} Hz",
              xlabel="time [s]", ylabel="amplitude")
    ax[0].grid(True)
    ax[1].plot(freqs[:half], mag[:half])
    ax[1].set(title="Magnitude spectrum (FFT)",
              xlabel="frequency [Hz]", ylabel="|X(f)| / N")
    ax[1].grid(True)
    _finish(fig, "task1_single_sinusoid")
    print(f"  Single tone peak should be at {f0:.0f} Hz "
          f"-> found at {freqs[:half][np.argmax(mag[:half])]:.1f} Hz")

    # --- (b) sum of two sinusoids --------------------------------- #
    f1, f2 = 80.0, 220.0
    x2 = np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)
    mag2 = np.abs(np.fft.fft(x2)) / N
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(freqs[:half], mag2[:half])
    ax.set(title=f"Sum of two sinusoids ({f1:.0f} Hz + {f2:.0f} Hz)",
           xlabel="frequency [Hz]", ylabel="|X(f)| / N")
    ax.grid(True)
    _finish(fig, "task1_two_sinusoids")

    # --- (c) effect of sampling rate fs on max observable frequency #
    # Max observable (Nyquist) frequency = fs/2.  Sampling a 300 Hz tone at
    # fs = 500 Hz (Nyquist 250 Hz) causes aliasing: it folds to fs - 300 = 200 Hz.
    f_tone = 300.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, fs_demo in enumerate([1000.0, 500.0]):
        nn = np.arange(N)
        tt = nn / fs_demo
        xx = np.cos(2 * np.pi * f_tone * tt)
        ff = np.fft.fftfreq(N, d=1 / fs_demo)
        mm = np.abs(np.fft.fft(xx)) / N
        h = N // 2
        ax[k].plot(ff[:h], mm[:h])
        ax[k].axvline(fs_demo / 2, color="r", ls="--", label="Nyquist fs/2")
        ax[k].set(title=f"fs = {fs_demo:.0f} Hz  (tone = {f_tone:.0f} Hz)",
                  xlabel="frequency [Hz]", ylabel="|X|/N")
        ax[k].legend(); ax[k].grid(True)
    _finish(fig, "task1_sampling_rate_aliasing")
    print("  fs=500 Hz < 2*300 Hz -> 300 Hz tone aliases to "
          f"{500 - 300:.0f} Hz (Nyquist violated).")

    # --- (d) effect of N on frequency resolution ------------------ #
    # Resolution  df = fs / N.   More samples -> finer bins -> closer tones
    # can be told apart.  Two tones 30 Hz apart:
    fa, fb = 100.0, 130.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, Ndemo in enumerate([64, 1024]):
        fs_d = 1000.0
        tt = np.arange(Ndemo) / fs_d
        xx = np.cos(2 * np.pi * fa * tt) + np.cos(2 * np.pi * fb * tt)
        ff = np.fft.fftfreq(Ndemo, d=1 / fs_d)
        mm = np.abs(np.fft.fft(xx)) / Ndemo
        h = Ndemo // 2
        ax[k].plot(ff[:h], mm[:h], marker=".")
        ax[k].set(title=f"N = {Ndemo}  ->  df = {fs_d / Ndemo:.1f} Hz",
                  xlabel="frequency [Hz]", ylabel="|X|/N", xlim=(0, 300))
        ax[k].grid(True)
    _finish(fig, "task1_frequency_resolution")
    print(f"  Two tones {fa:.0f}/{fb:.0f} Hz: resolved with large N, "
          "blurred with small N (df = fs/N).")


# ================================================================== #
#  TASK 2 :  Digital symbols (BPSK / QPSK)                           #
# ================================================================== #
def task2_digital_symbols():
    print("\n=== Task 2: Digital symbols (BPSK / QPSK) ===")

    # --- BPSK round-trip check ------------------------------------ #
    bits = random_bits(16)
    bpsk = bpsk_modulate(bits)
    rec = bpsk_demodulate(bpsk)
    assert np.array_equal(bits, rec), "BPSK round-trip failed!"
    print(f"  BPSK round-trip OK   (bits = {bits.tolist()})")

    # --- QPSK round-trip check ------------------------------------ #
    bits2 = random_bits(16)
    qpsk = qpsk_modulate(bits2)
    rec2 = qpsk_demodulate(qpsk)
    assert np.array_equal(bits2, rec2), "QPSK round-trip failed!"
    print(f"  QPSK round-trip OK   (symbol power = "
          f"{np.mean(np.abs(qpsk) ** 2):.3f}, should be 1.0)")

    # --- Constellation diagrams ----------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

    ax[0].scatter(np.real(bpsk), np.imag(bpsk), s=80, color="C0", zorder=3)
    ax[0].set(title="BPSK constellation", xlabel="In-phase (I)",
              ylabel="Quadrature (Q)", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
    ax[0].grid(True); ax[0].set_aspect("equal")

    # label the four QPSK points with their bit pairs
    labels = {"00": (1 + 1j), "01": (-1 + 1j), "11": (-1 - 1j), "10": (1 - 1j)}
    for lab, c in labels.items():
        p = c * _INV_SQRT2
        ax[1].scatter(p.real, p.imag, s=80, color="C1", zorder=3)
        ax[1].annotate(lab, (p.real, p.imag),
                       textcoords="offset points", xytext=(8, 8))
    ax[1].set(title="QPSK constellation", xlabel="In-phase (I)",
              ylabel="Quadrature (Q)", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
    ax[1].grid(True); ax[1].set_aspect("equal")
    _finish(fig, "task2_constellations")


# ================================================================== #
#  TASK 3 :  Building a simple OFDM symbol (ideal channel)           #
# ================================================================== #
def task3_basic_ofdm():
    print("\n=== Task 3: Basic OFDM symbol (ideal channel) ===")

    N = 64                            # FFT size
    active = np.arange(1, 17)         # 16 active subcarriers (bins 1..16)
    n_active = len(active)

    for scheme, mod, demod, bits_per_sym in [
        ("BPSK", bpsk_modulate, bpsk_demodulate, 1),
        ("QPSK", qpsk_modulate, qpsk_demodulate, 2),
    ]:
        bits = random_bits(n_active * bits_per_sym)
        symbols = mod(bits)

        # --- transmitter --- #
        tx_time = ofdm_modulate(symbols, N, active)

        # --- ideal channel: received == transmitted --- #
        rx_time = tx_time

        # --- receiver --- #
        rx_symbols = ofdm_demodulate(rx_time, N, active)
        rx_bits = demod(rx_symbols)

        bit_err = ber(bits, rx_bits)
        print(f"  {scheme}: BER over ideal channel = {bit_err:.0e} "
              f"(expected 0).  Recovered symbols match: "
              f"{np.allclose(rx_symbols, symbols)}")

        if scheme == "QPSK":
            # plot the recovered constellation for QPSK (perfect dots)
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
            tx_t = np.real(tx_time)
            ax[0].plot(tx_t)
            ax[0].set(title="OFDM time-domain symbol (real part)",
                      xlabel="sample index n", ylabel="amplitude")
            ax[0].grid(True)
            ax[1].scatter(np.real(rx_symbols), np.imag(rx_symbols),
                          color="C1", zorder=3)
            ax[1].set(title="Recovered QPSK subcarriers (ideal)",
                      xlabel="I", ylabel="Q", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
            ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
            ax[1].grid(True); ax[1].set_aspect("equal")
            _finish(fig, "task3_ofdm_ideal")


# ================================================================== #
#  TASK 4 :  Adding noise (AWGN, BER vs SNR)                         #
# ================================================================== #
def task4_noise():
    print("\n=== Task 4: Adding noise (AWGN) ===")

    N = 64
    active = np.arange(1, 17)
    n_active = len(active)

    # --- (a) constellations before / after noise (QPSK) ----------- #
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for col, snr_db in enumerate([np.inf, 20, 5]):
        # accumulate many symbols so the cloud is visible
        pts = []
        for _ in range(200):
            bits = random_bits(n_active * 2)
            tx = ofdm_modulate(qpsk_modulate(bits), N, active)
            if np.isinf(snr_db):
                rx = tx
            else:
                rx, _ = add_awgn(tx, snr_db)
            pts.append(ofdm_demodulate(rx, N, active))
        pts = np.concatenate(pts)
        title = "No noise" if np.isinf(snr_db) else f"SNR = {snr_db} dB"
        ax[col].scatter(np.real(pts), np.imag(pts), s=6, alpha=0.3, color="C1")
        ax[col].set(title=title, xlabel="I", ylabel="Q",
                    xlim=(-2, 2), ylim=(-2, 2))
        ax[col].axhline(0, color="k", lw=0.5); ax[col].axvline(0, color="k", lw=0.5)
        ax[col].grid(True); ax[col].set_aspect("equal")
    _finish(fig, "task4_noisy_constellations")
    print("  Lower SNR -> received symbols spread further from ideal points.")

    # --- (b) BER vs SNR curve ------------------------------------- #
    # Only 16 of the 64 bins carry data, so each subcarrier sees 64/16 = 6 dB
    # more SNR than the time-domain SNR on this axis. The range is chosen so
    # the whole BER curve (waterfall) is visible.
    snr_range = np.arange(-10, 7, 1.0)
    n_symbols = 3000                       # OFDM symbols averaged per SNR point

    results = {}
    for scheme, mod, demod, bps in [
        ("BPSK", bpsk_modulate, bpsk_demodulate, 1),
        ("QPSK", qpsk_modulate, qpsk_demodulate, 2),
    ]:
        ber_sim = []
        for snr_db in snr_range:
            errors = 0
            total = 0
            for _ in range(n_symbols):
                bits = random_bits(n_active * bps)
                tx = ofdm_modulate(mod(bits), N, active)
                rx, _ = add_awgn(tx, snr_db)
                rx_bits = demod(ofdm_demodulate(rx, N, active))
                errors += np.sum(bits != rx_bits)
                total += len(bits)
            ber_sim.append(errors / total)
        results[scheme] = np.array(ber_sim)

    # --- theoretical reference curves ----------------------------- #
    # signal power is deterministic here: n_active unit-power symbols spread
    # over N time samples -> P_sig = n_active / N**2 .  After the FFT each
    # active subcarrier sees complex noise of variance N*noise_power, i.e.
    # real-axis variance sigma^2 = N*noise_power/2.
    snr_lin = 10 ** (snr_range / 10)
    p_sig = n_active / N ** 2
    noise_power = p_sig / snr_lin
    sigma_real = np.sqrt(N * noise_power / 2)
    # BPSK real amplitude = 1 ; QPSK real amplitude = 1/sqrt(2)
    ber_bpsk_th = 0.5 * erfc((1.0) / (sigma_real * np.sqrt(2)))
    ber_qpsk_th = 0.5 * erfc((_INV_SQRT2) / (sigma_real * np.sqrt(2)))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(snr_range, results["BPSK"], "o-", color="C0", label="BPSK (sim)")
    ax.semilogy(snr_range, ber_bpsk_th, "--", color="C0", label="BPSK (theory)")
    ax.semilogy(snr_range, results["QPSK"], "s-", color="C1", label="QPSK (sim)")
    ax.semilogy(snr_range, ber_qpsk_th, "--", color="C1", label="QPSK (theory)")
    ax.set(title="OFDM BER vs SNR (AWGN)",
           xlabel="SNR [dB]  (time-domain, per activity sheet)",
           ylabel="Bit Error Rate", ylim=(1e-5, 1))
    ax.grid(True, which="both"); ax.legend()
    _finish(fig, "task4_ber_vs_snr")

    print("  BER decreases as SNR rises (simulation tracks theory):")
    for i, s in enumerate(snr_range):
        if int(s) % 4 == 0:
            print(f"    SNR={s:5.1f} dB  BPSK={results['BPSK'][i]:.2e}  "
                  f"QPSK={results['QPSK'][i]:.2e}")


# ================================================================== #
#  TASK 5 :  Cyclic prefix and simple multipath                     #
# ================================================================== #
def add_cp(symbol, cp_len):
    """Prepend a cyclic prefix: copy the last cp_len samples to the front."""
    return np.concatenate([symbol[-cp_len:], symbol])


def task5_cyclic_prefix():
    print("\n=== Task 5: Cyclic prefix and multipath ===")

    N = 64
    cp_len = 16
    active = np.arange(1, 17)
    n_active = len(active)
    M = 2000                       # number of OFDM symbols in the stream
    bps = 2                        # QPSK

    def run_stream(h, use_cp):
        """Transmit M OFDM symbols through multipath channel h (NO noise, so the
        only impairment is the multipath itself).  Returns BER, mean EVM, and the
        recovered subcarriers of one mid-stream symbol.  The only difference
        between the two runs is the cyclic prefix, so any difference is its doing.
        """
        H = np.fft.fft(h, N)                       # channel frequency response
        all_bits, all_symbols, all_ideal = [], [], []
        for _ in range(M):
            bits = random_bits(n_active * bps)
            sym_f = qpsk_modulate(bits)
            sym = ofdm_modulate(sym_f, N, active)
            if use_cp:
                sym = add_cp(sym, cp_len)
            all_bits.append(bits)
            all_symbols.append(sym)
            all_ideal.append(sym_f)

        tx_stream = np.concatenate(all_symbols)
        rx_stream = np.convolve(tx_stream, h)      # multipath channel

        block = N + cp_len if use_cp else N
        errors, total, evm = 0, 0, []
        mid_const = None
        for k in range(M):
            seg = rx_stream[k * block:(k + 1) * block]
            if use_cp:
                seg = seg[cp_len:]                  # remove cyclic prefix
            seg = seg[:N]
            Y = np.fft.fft(seg)
            X_hat = Y[active] / H[active]           # one-tap equaliser
            errors += np.sum(all_bits[k] != qpsk_demodulate(X_hat))
            total += len(all_bits[k])
            evm.append(np.mean(np.abs(X_hat - all_ideal[k]) ** 2))
            if k == M // 2:
                mid_const = X_hat
        return errors / total, float(np.mean(evm)), mid_const

    # Two channels: (1) the sheet's mild echo, (2) a stronger/longer echo.
    channels = {
        "h = [1, 0, 0.4]  (sheet example, mild)": np.array([1.0, 0.0, 0.4]),
        "h = [1,0,0,0,0,0.7]  (stronger, delay 5)":
            np.array([1.0, 0, 0, 0, 0, 0.7]),
    }

    fig, axes = plt.subplots(len(channels), 2, figsize=(9, 8))
    for row, (name, h) in enumerate(channels.items()):
        ber_no, evm_no, const_no = run_stream(h, use_cp=False)
        ber_yes, evm_yes, const_yes = run_stream(h, use_cp=True)
        print(f"\n  Channel: {name}   memory = {len(h) - 1} samples (CP = {cp_len})")
        print(f"    WITHOUT CP : BER = {ber_no:.3e}   EVM = {evm_no:.4f}")
        print(f"    WITH    CP : BER = {ber_yes:.3e}   EVM = {evm_yes:.4f}")

        a0, a1 = axes[row]
        a0.scatter(const_no.real, const_no.imag, s=10, alpha=0.4, color="C3")
        a0.set(title=f"No CP\nBER={ber_no:.1e}, EVM={evm_no:.3f}",
               xlabel="I", ylabel="Q", xlim=(-2, 2), ylim=(-2, 2))
        a1.scatter(const_yes.real, const_yes.imag, s=10, alpha=0.4, color="C2")
        a1.set(title=f"With CP\nBER={ber_yes:.1e}, EVM={evm_yes:.3f}",
               xlabel="I", ylabel="Q", xlim=(-2, 2), ylim=(-2, 2))
        for a in (a0, a1):
            a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
            a.grid(True); a.set_aspect("equal")
        a0.text(-1.9, 1.6, name.split("  ")[0], fontsize=8)

    print("\n  Even a mild echo distorts the subcarriers (EVM > 0) without a CP;")
    print("  a stronger echo turns that distortion into actual bit errors.")
    print("  Adding a cyclic prefix longer than the channel memory turns the")
    print("  channel into a clean per-subcarrier multiply -> EVM ~ 0, BER = 0")
    print("  after one-tap equalisation.  (All runs here are noise-free, so the")
    print("  only impairment is the multipath.)")
    _finish(fig, "task5_cyclic_prefix")


# ================================================================== #
#  TASK 6 :  Frequency offset                                       #
# ================================================================== #
def apply_frequency_offset(samples, f_offset, fs):
    """Multiply samples by a rotating complex exponential exp(j 2 pi f n / fs)."""
    n = np.arange(len(samples))
    return samples * np.exp(1j * 2 * np.pi * f_offset * n / fs)


def task6_frequency_offset():
    print("\n=== Task 6: Frequency offset ===")

    N = 64
    active = np.arange(1, 17)
    n_active = len(active)
    fs = 1e6                        # sampling rate [Hz]

    bits = random_bits(n_active * 2)
    symbols = qpsk_modulate(bits)
    tx = ofdm_modulate(symbols, N, active)

    # small and larger offsets (as a fraction of the subcarrier spacing fs/N)
    df_sub = fs / N                 # subcarrier spacing
    offsets = [0.0, 0.02 * df_sub, 0.3 * df_sub]   # 0, small, larger

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for col, foff in enumerate(offsets):
        rx = apply_frequency_offset(tx, foff, fs)
        rx_sym = ofdm_demodulate(rx, N, active)
        ax[col].scatter(np.real(rx_sym), np.imag(rx_sym), color="C4", zorder=3)
        frac = foff / df_sub
        ax[col].set(title=f"offset = {foff:.0f} Hz  ({frac:.2f} x subcarrier)",
                    xlabel="I", ylabel="Q", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
        ax[col].axhline(0, color="k", lw=0.5); ax[col].axvline(0, color="k", lw=0.5)
        ax[col].grid(True); ax[col].set_aspect("equal")
    _finish(fig, "task6_frequency_offset")
    print("  Small offset -> the whole constellation rotates.")
    print("  Larger offset -> subcarriers leak into each other (ICI), so the")
    print("  points also smear out, not just rotate.")

    # --- optional extension: correction by counter-rotation -------- #
    # 0.3 x spacing rotates QPSK past 45 deg, so errors appear before correction
    foff = 0.3 * df_sub
    rx = apply_frequency_offset(tx, foff, fs)
    rx_corrected = apply_frequency_offset(rx, -foff, fs)   # opposite rotation
    rx_sym_corr = ofdm_demodulate(rx_corrected, N, active)
    err_before = ber(bits, qpsk_demodulate(ofdm_demodulate(rx, N, active)))
    err_after = ber(bits, qpsk_demodulate(rx_sym_corr))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    rx_sym = ofdm_demodulate(rx, N, active)
    ax[0].scatter(np.real(rx_sym), np.imag(rx_sym), color="C3", zorder=3)
    ax[0].set(title=f"Before correction (BER={err_before:.2e})",
              xlabel="I", ylabel="Q", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    ax[1].scatter(np.real(rx_sym_corr), np.imag(rx_sym_corr),
                  color="C2", zorder=3)
    ax[1].set(title=f"After correction (BER={err_after:.2e})",
              xlabel="I", ylabel="Q", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    for a in ax:
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.grid(True); a.set_aspect("equal")
    _finish(fig, "task6_offset_correction")
    print(f"  Optional correction: BER {err_before:.2e} -> {err_after:.2e} "
          "(works because the offset is known).")


# ================================================================== #
#  Main                                                             #
# ================================================================== #
def main():
    print("=" * 60)
    print(" ExA Part 1 - Simulation-Based OFDM")
    print("=" * 60)
    task1_signals_and_fft()
    task2_digital_symbols()
    task3_basic_ofdm()
    task4_noise()
    task5_cyclic_prefix()
    task6_frequency_offset()
    print("\nAll Part 1 tasks completed.")
    if not SAVE_FIGS:
        plt.show()


if __name__ == "__main__":
    main()
