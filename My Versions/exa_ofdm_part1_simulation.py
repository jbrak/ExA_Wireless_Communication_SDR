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
    # np.real(symbols)>0 extracts real part (for complex input) and compares it to 0
    # astype(int) converts the boolean array to integers (True->1, False->0)
    # thus returning a sequence of bits corresponding to the BPSK symbols.

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

    # number of bits must be even, since each QPSK symbol is made from 2 bits
    assert len(bits) % 2 == 0, "QPSK needs an even number of bits"
    # Reshapes the bits array into a 2D array with 2 columns (one for each bit of the pair). Each row will contain a pair of bits (b0, b1) for one QPSK symbol.
    # -1 means number of rows is inferred from the length of bits, and 2 means there are 2 columns.
    b = bits.reshape(-1, 2)

    # split into seperate arrays for the two bits of each pair
    b0, b1 = b[:, 0], b[:, 1]

    # compute the real and imaginary parts of the QPSK symbols based on the bit values
    i = 1 - 2 * b1
    q = 1 - 2 * b0

    # combine the real and imaginary parts into complex symbols and normalise the power
    return (i + 1j * q) * _INV_SQRT2


def qpsk_demodulate(symbols):
    """Nearest-quadrant decision for QPSK; returns the recovered bit array."""

    # converts symbols to a numpy array
    s = np.asarray(symbols)

    # determine the bits based on the sign of the real and imaginary parts of the symbols
    b1 = (np.real(s) < 0).astype(int)   # real<0 -> b1=1
    b0 = (np.imag(s) < 0).astype(int)   # imag<0 -> b0=1

    # combine the two bit arrays into a single array of bits, preserving the order of the original bit pairs
    # column_stack stacks the two arrays as columns, and reshape(-1) flattens the 2D array back into a 1D array of bits.
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
    # define an array of length N filled with complex zeros to represent the frequency-domain representation of the OFDM symbol.
    # Each element corresponds to a frequency bin (subcarrier) in the FFT.
    # Each element is initialized to zero, indicating that no modulation symbols are assigned to those bins yet.
    # Each bin corresponds to a specific frequncy in the ODFM system, and the active subcarriers will be filled with the provided modulation symbols.
    X = np.zeros(N, dtype=complex)   # frequency-domain array, all bins zero

    # assign the provided modulation symbols to the specified active subcarriers in the frequency-domain array.
    X[active] = symbols              # fill the active subcarriers

    # perform the inverse fast Fourier transform (IFFT) on the frequency-domain array to convert it to the time-domain representation of the OFDM symbol.
    # The IFFT takes the complex frequency-domain data and generates a time-domain waveform that can be transmitted over the channel.
    # Ensures that the time-domain signal is of length N, matching the FFT size.
    # Ensures that the time-domain subcarriers remain orthogonal, preventing interference between them.
    return np.fft.ifft(X)            # IFFT -> time-domain waveform


def ofdm_demodulate(rx_time, N, active):
    """FFT the received block and extract the active subcarriers."""
    # Forwards fourier transform (FFT) of the received time-domain OFDM symbol to convert it back to the frequency domain.
    Y = np.fft.fft(rx_time)

    # Extract the values of the active subcarriers from the frequency-domain representation of the received OFDM symbol.
    return Y[active]


# ---- AWGN ------------------------------------------------------------------ #
def add_awgn(signal, snr_db):
    """
    Add complex AWGN at the requested SNR (dB).  Noise power is referenced to
    the measured average power of `signal`, as in the activity sheet:
        noise_power = signal_power / 10**(snr_db/10)
    The complex noise is split equally between real and imaginary parts.
    """

    # Calculates the average power of the input signal by taking the mean of the squared magnitudes of its complex values.
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculates the noise power based on the desired SNR in dB. The formula converts the SNR from dB to a linear scale
    # and divides the signal power by this value to determine the appropriate noise power.
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generates complex additive white Gaussian noise (AWGN) with the calculated noise power.
    # The noise is created by generating two independent standard normal random variables
    # (one for the real part and one for the imaginary part) and scaling them to achieve the desired noise power.
    noise = np.sqrt(noise_power / 2) * (
        RNG.standard_normal(signal.shape) + 1j * RNG.standard_normal(signal.shape)
    )

    # returns the noisy signal (original signal plus noise) and
    # the calculated noise power for further analysis or verification.
    return signal + noise, noise_power


def ber(tx_bits, rx_bits):
    """Bit error rate between two equal-length bit arrays."""

    # Ensures the comparison only applies to overlapping bits if the lengths differ, preventing index errors.
    n = min(len(tx_bits), len(rx_bits))

    # Calculates the number of bit errors by comparing the transmitted bits (tx_bits) and received bits (rx_bits) up to the length n.
    # The expression tx_bits[:n] != rx_bits[:n] creates a boolean array where True indicates a bit error (mismatch) and False indicates a match.
    # Sum all the True values (bit errors) and divide by n to get the bit error rate (BER), which is the fraction of bits that were received incorrectly.
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

    # Define a sinusiod A * cos(2 * pi * f0 * t) with amplitude
    # A and frequency f0
    x = A * np.cos(2 * np.pi * f0 * t)

    # compute the fast fourier transform (FFT) of the signal x
    X = np.fft.fft(x)

    # compute the corresponding frequency bins for the FFT output
    freqs = np.fft.fftfreq(N, d=1 / fs)          # bin centre frequencies

    # compute the magnitude spectrum of the FFT output and
    # normalise it by the number of samples N
    mag = np.abs(X) / N                          # normalise magnitude

    # plot only the positive half of the spectrum (real signal -> symmetric)
    half = N // 2

    # create a figure with two subplots: one for the time-domain
    # signal and one for the magnitude spectrum
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # plot the time-domain signal (first 200 samples) in the first subplot
    ax[0].plot(t[:200], x[:200])
    ax[0].set(title=f"Single sinusoid  f = {f0:.0f} Hz",
              xlabel="time [s]", ylabel="amplitude")
    ax[0].grid(True)

    # plot the magnitude spectrum (first half of the FFT) in the second subplot
    ax[1].plot(freqs[:half], mag[:half])
    ax[1].set(title="Magnitude spectrum (FFT)",
              xlabel="frequency [Hz]", ylabel="|X(f)| / N")
    ax[1].grid(True)
    _finish(fig, "task1_single_sinusoid")

    # print the expected and found peak frequencies in the magnitude spectrum
    print(f"  Single tone peak should be at {f0:.0f} Hz "
          f"-> found at {freqs[:half][np.argmax(mag[:half])]:.1f} Hz")

    # --- (b) sum of two sinusoids --------------------------------- #

    # Define two frequencies f1 and f2 for the sum of two sinusoids
    f1, f2 = 80.0, 220.0

    # define the sum of two sinusoids with different amplitudes and frequencies
    x2 = np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)

    # compute the magnitude spectrum of the sum of two sinusoids and
    # normalise it by the number of samples N
    mag2 = np.abs(np.fft.fft(x2)) / N

    # plot the magnitude spectrum of the sum of two sinusoids (first half of the FFT)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(freqs[:half], mag2[:half])
    ax.set(title=f"Sum of two sinusoids ({f1:.0f} Hz + {f2:.0f} Hz)",
           xlabel="frequency [Hz]", ylabel="|X(f)| / N")
    ax.grid(True)

    _finish(fig, "task1_two_sinusoids")

    # --- (c) effect of sampling rate fs on max observable frequency #
    # Max observable (Nyquist) frequency = fs/2.  Sampling a 300 Hz tone at
    # fs = 500 Hz (Nyquist 250 Hz) causes aliasing: it folds to fs - 300 = 200 Hz.

    # set frequency of the tone to be sampled
    f_tone = 300.0

    # create a figure with two subplots to compare the effect of
    # different sampling rates on the frequency spectrum
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    # loop over two different sampling rates (1000 Hz and 500 Hz) to
    # demonstrate the effect of aliasing
    for k, fs_demo in enumerate([1000.0, 500.0]):

        # create a time axis for the given sampling rate and number of samples
        nn = np.arange(N)
        tt = nn / fs_demo

        # define a cosine wave at the specified tone frequency
        xx = np.cos(2 * np.pi * f_tone * tt)

        # fast fourier transform of the cosine wave and compute the
        # corresponding frequency bins
        ff = np.fft.fftfreq(N, d=1 / fs_demo)
        mm = np.abs(np.fft.fft(xx)) / N

        # real signal -> symmetric spectrum, so only plot the first half of the FFT
        h = N // 2

        # plot the magnitude spectrum for the current sampling rate in the
        # corresponding subplot
        ax[k].plot(ff[:h], mm[:h])
        ax[k].axvline(fs_demo / 2, color="r", ls="--", label="Nyquist fs/2")
        ax[k].set(title=f"fs = {fs_demo:.0f} Hz  (tone = {f_tone:.0f} Hz)",
                  xlabel="frequency [Hz]", ylabel="|X|/N")
        ax[k].legend(); ax[k].grid(True)
    _finish(fig, "task1_sampling_rate_aliasing")

    # print a message explaining the aliasing effect observed when sampling a
    # 300 Hz tone at 500 Hz, which violates the Nyquist criterion (fs < 2*f0)
    print("  fs=500 Hz < 2*300 Hz -> 300 Hz tone aliases to "
          f"{500 - 300:.0f} Hz (Nyquist violated).")

    # --- (d) effect of N on frequency resolution ------------------ #
    # Resolution  df = fs / N.   More samples -> finer bins -> closer tones
    # can be told apart.  Two tones 30 Hz apart:

    # define two frequencies fa and fb that are 30 Hz apart
    # to demonstrate the effect of frequency resolution on the
    # ability to distinguish between closely spaced tones
    fa, fb = 100.0, 130.0

    # define 2 plots to compare the effect of different numbers of
    # samples (N) on the frequency resolution
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    # loop over two different numbers of samples (64 and 1024)
    for k, Ndemo in enumerate([64, 1024]):
        # define a sampling rate for the demonstration (1000 Hz)
        fs_d = 1000.0

        # create a time axis for the given number of samples and sampling rate
        tt = np.arange(Ndemo) / fs_d

        # define a signal that is the sum of two cosine waves at frequencies fa and fb
        xx = np.cos(2 * np.pi * fa * tt) + np.cos(2 * np.pi * fb * tt)

        # compute the corresponding frequency bins for the FFT output
        ff = np.fft.fftfreq(Ndemo, d=1 / fs_d)
        mm = np.abs(np.fft.fft(xx)) / Ndemo

        # real signal -> symmetric spectrum, so only plot the first half of the FFT
        h = Ndemo // 2

        # plot the magnitude spectrum for the current number of samples in
        # the corresponding subplot
        ax[k].plot(ff[:h], mm[:h], marker=".")
        ax[k].set(title=f"N = {Ndemo}  ->  df = {fs_d / Ndemo:.1f} Hz",
                  xlabel="frequency [Hz]", ylabel="|X|/N", xlim=(0, 300))
        ax[k].grid(True)
    _finish(fig, "task1_frequency_resolution")

    # print a message explaining the effect of the number of samples
    # (N) on the frequency resolution (df = fs/N)
    print(f"  Two tones {fa:.0f}/{fb:.0f} Hz: resolved with large N, "
          "blurred with small N (df = fs/N).")


# ================================================================== #
#  TASK 2 :  Digital symbols (BPSK / QPSK)                           #
# ================================================================== #
def task2_digital_symbols():
    print("\n=== Task 2: Digital symbols (BPSK / QPSK) ===")

    # --- BPSK round-trip check ------------------------------------ #
    # Generate random bits, modulate them using BPSK, demodulate the symbols back to bits,
    # and check if the original bits match the recovered bits.
    bits = random_bits(16)
    bpsk = bpsk_modulate(bits)
    rec = bpsk_demodulate(bpsk)
    assert np.array_equal(bits, rec), "BPSK round-trip failed!"
    print(f"  BPSK round-trip OK   (bits = {bits.tolist()})")

    # --- QPSK round-trip check ------------------------------------ #
    # Generate random bits, modulate them using QPSK, demodulate the symbols back to bits,
    # and check if the original bits match the recovered bits.
    # double check signal power is 1
    bits2 = random_bits(16)
    qpsk = qpsk_modulate(bits2)
    rec2 = qpsk_demodulate(qpsk)
    assert np.array_equal(bits2, rec2), "QPSK round-trip failed!"
    print(f"  QPSK round-trip OK   (symbol power = "
          f"{np.mean(np.abs(qpsk) ** 2):.3f}, should be 1.0)")

    # --- Constellation diagrams ----------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

    # plot each BPSK point as a blue dot in the constellation diagram
    ax[0].scatter(np.real(bpsk), np.imag(bpsk), s=80, color="C0", zorder=3)
    ax[0].set(title="BPSK constellation", xlabel="In-phase (I)",
              ylabel="Quadrature (Q)", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
    ax[0].grid(True); ax[0].set_aspect("equal")

    # label the four QPSK points with their bit pairs
    labels = {"00": (1 + 1j), "01": (-1 + 1j), "11": (-1 - 1j), "10": (1 - 1j)}

    # loop through each label and its corresponding complex point, scale it by 1/sqrt(2)
    # for power normalization, and plot it in the constellation diagram
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

    # Define the total number of frequncy bins
    # i.e. the total numbher of subcarriers in the OFDM system.
    # The FFT size N determines the number of frequency bins available for modulation.
    N = 64                            # FFT size

    # Define the indices of the active subcarriers (bins 1 to 16)
    active = np.arange(1, 17)         # 16 active subcarriers (bins 1..16)

    # Define the number of active subcarriers based on the length of the active array
    n_active = len(active)

    # Loop through different modulation schemes (BPSK and QPSK) to test the OFDM system with each scheme.
    for scheme, mod, demod, bits_per_sym in [
        ("BPSK", bpsk_modulate, bpsk_demodulate, 1),
        ("QPSK", qpsk_modulate, qpsk_demodulate, 2),
    ]:

        # Generate random bits for the active subcarriers based on the modulation scheme's bits per symbol.
        bits = random_bits(n_active * bits_per_sym)

        # Modulate the random bits into symbols using the specified modulation scheme (BPSK or QPSK).
        symbols = mod(bits)

        # --- transmitter --- #
        # Modulate the symbols into a time-domain OFDM signal using the IFFT, placing the symbols on the specified active subcarriers.
        tx_time = ofdm_modulate(symbols, N, active)

        # --- ideal channel: received == transmitted --- #

        # since this is an ideal channel,
        # the received time-domain signal is the same as the transmitted signal (no noise or distortion).
        rx_time = tx_time

        # --- receiver --- #
        # Demodulate the received time-domain OFDM signal back into symbols using the FFT,
        # extracting the values of the active subcarriers.
        rx_symbols = ofdm_demodulate(rx_time, N, active)
        # Demodulate the received symbols back into bits using the specified demodulation scheme (BPSK or QPSK).
        rx_bits = demod(rx_symbols)

        # Calculate the bit error rate (BER) by comparing the original transmitted bits with the received bits.
        bit_err = ber(bits, rx_bits)

        # print the scheme, the calaculate BER (0), and whether the recovered symbols match the original symbols.
        print(f"  {scheme}: BER over ideal channel = {bit_err:.0e} "
              f"(expected 0).  Recovered symbols match: "
              f"{np.allclose(rx_symbols, symbols)}")

        # For QPSK, plot the recovered constellation diagram to visualize the received symbols in the complex plane.
        # Also plot the time-domain OFDM symbol (real part) to show the waveform of the transmitted signal.
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

    # Define the total number of frequncy bins
    N = 64

    # Define 16 active subcarriers (bins 1 to 16) for the OFDM system.
    active = np.arange(1, 17)
    n_active = len(active)

    # --- (a) constellations before / after noise (QPSK) ----------- #

    # Define a plot with three subplots to visualize the effect of noise on the received
    # QPSK symbols at different SNR levels.
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Loop through different SNR levels (in dB) to simulate the effect of noise on the received symbols.
    for col, snr_db in enumerate([np.inf, 20, 5]):
        # accumulate many symbols so the cloud is visible

        # Define an empty list to store the received symbols for the current SNR level.
        pts = []

        # Loop over 200 OFDM symbols to generate a cloud of received symbols for the current SNR level.
        for _ in range(200):
            # generate random bits for the active subcarriers (2 bits per symbol for QPSK)
            bits = random_bits(n_active * 2)

            # Work out the transmited time domain ODFM signal
            # Modulate the random bits into QPSK symbols, then modulate those symbols into a time-domain OFDM signal
            # using the IFFT, placing the symbols on the specified active subcarriers.
            tx = ofdm_modulate(qpsk_modulate(bits), N, active)

            # Add noise to the transmitted signal based on the current SNR level.
            # If the SNR is infinite (no noise), the received signal is the same as the transmitted signal.
            # Otherwise, add complex AWGN to the transmitted signal.
            if np.isinf(snr_db):
                rx = tx
            else:
                # Add complex AWGN to the transmitted signal based on the current SNR level.
                rx, _ = add_awgn(tx, snr_db)
            # Append the demodulated received symbols (from the active subcarriers) to the list of received points for the current SNR level.
            pts.append(ofdm_demodulate(rx, N, active))

        # concatenate the list of received points into a single array for plotting.
        pts = np.concatenate(pts)

        # Set the title of the subplot based on the current SNR level.
        title = "No noise" if np.isinf(snr_db) else f"SNR = {snr_db} dB"

        # Plot the received symbols in the complex plane (I vs Q) for the current SNR level.
        ax[col].scatter(np.real(pts), np.imag(pts), s=6, alpha=0.3, color="C1")
        ax[col].set(title=title, xlabel="I", ylabel="Q",
                    xlim=(-2, 2), ylim=(-2, 2))
        ax[col].axhline(0, color="k", lw=0.5); ax[col].axvline(0, color="k", lw=0.5)
        ax[col].grid(True); ax[col].set_aspect("equal")
    _finish(fig, "task4_noisy_constellations")

    # print a message explaining the effect of noise on the received symbols at different SNR levels.
    print("  Lower SNR -> received symbols spread further from ideal points.")

    # --- (b) BER vs SNR curve ------------------------------------- #
    # Only 16 of the 64 bins carry data, so each subcarrier sees 64/16 = 6 dB
    # more SNR than the time-domain SNR on this axis. The range is chosen so
    # the whole BER curve (waterfall) is visible.

    # 16 bins carry data bat all 64 bins see the same noise power
    # Thus each subcarrier sees 16/64 = 1/4 of of the noise power
    # This results in linear SNR per subcarrier being 4 times the linear SNR in the time domain
    # Thus the SNR per subcarrier in dB is 10*log10(4) = 6.02 dB higher than the time domain SNR


    # Define the range of SNR values (in dB) to simulate for the BER vs SNR curve.
    snr_range = np.arange(-10, 7, 1.0)

    # Define the number of OFDM symbols to average for each SNR point in the simulation.
    n_symbols = 3000                       # OFDM symbols averaged per SNR point

    # Define a dictionary to store the simulated BER results f
    # or each modulation scheme (BPSK and QPSK).
    results = {}

    # Loop through different modulation schemes (BPSK and QPSK)
    # to simulate the BER vs SNR curve for each scheme.
    for scheme, mod, demod, bps in [
        ("BPSK", bpsk_modulate, bpsk_demodulate, 1),
        ("QPSK", qpsk_modulate, qpsk_demodulate, 2),
    ]:
        # Initialize an empty list to store the simulated BER values for the current modulation scheme.
        ber_sim = []

        # Loop through each SNR value in the defined range to simulate the BER for that SNR level.
        for snr_db in snr_range:

            # Define variables to accumulate the total number
            # of bit errors and the total number of bits transmitted for the current SNR level.
            errors = 0
            total = 0

            # Loop through the specified number of OFDM symbols to
            # simulate the transmission and reception of bits for the current SNR level.
            for _ in range(n_symbols):
                # Generate random bits for the
                # active subcarriers based on the modulation scheme's bits per symbol.
                bits = random_bits(n_active * bps)

                # Modulate the random bits into a time-domain OFDM signal using the specified modulation scheme (BPSK or QPSK)
                # and the IFFT, placing the symbols on the specified active subcarriers.
                tx = ofdm_modulate(mod(bits), N, active)
                # Add noise to the transmitted signal based on the
                # current SNR level using the add_awgn function.
                rx, _ = add_awgn(tx, snr_db)

                # Demodulate the received time-domain OFDM signal back into symbols using the FFT,
                rx_bits = demod(ofdm_demodulate(rx, N, active))

                # Calculate the number of bit errors by comparing the original transmitted
                # bits with the received bits and accumulate the total number of bits transmitted.
                errors += np.sum(bits != rx_bits)
                total += len(bits)
            ber_sim.append(errors / total)
        results[scheme] = np.array(ber_sim)

    # --- theoretical reference curves ----------------------------- #
    # signal power is deterministic here: n_active unit-power symbols spread
    # over N time samples -> P_sig = n_active / N**2 .  After the FFT each
    # active subcarrier sees complex noise of variance N*noise_power, i.e.
    # real-axis variance sigma^2 = N*noise_power/2.

    # Convert the SNR values from dB to linear scale for theoretical calculations.
    snr_lin = 10 ** (snr_range / 10)

    # Calculate the signal power based on the number
    # of active subcarriers and the total number of frequency bins.
    # Signal power is deterministic. a known number of active symbols with power 1
    # are spread over N time samples, resulting in a signal power of n_active / N^2.
    p_sig = n_active / N ** 2

    # Calculate the noise power based on the signal power and the linear SNR values.
    noise_power = p_sig / snr_lin

    # Calculate the standard deviation of the
    # real part of the noise for theoretical BER calculations.
    sigma_real = np.sqrt(N * noise_power / 2)

    # BPSK real amplitude = 1 ; QPSK real amplitude = 1/sqrt(2)
    # generate the theoretical BER values for BPSK and QPSK
    # using the complementary error function (erfc) (i.e. 1-erf).
    # Derived from expressions for the probability that the guassian white noise
    # will push the received symbol across the decision boundary.
    ber_bpsk_th = 0.5 * erfc((1.0) / (sigma_real * np.sqrt(2)))
    ber_qpsk_th = 0.5 * erfc((_INV_SQRT2) / (sigma_real * np.sqrt(2)))


    # Create a plot to visualize the simulated and
    # theoretical BER vs SNR curves for both BPSK and QPSK modulation schemes.
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Plot the simulated BER values for BPSK and QPSK using markers and solid lines.
    ax.semilogy(snr_range, results["BPSK"], "o-", color="C0", label="BPSK (sim)")
    ax.semilogy(snr_range, ber_bpsk_th, "--", color="C0", label="BPSK (theory)")
    ax.semilogy(snr_range, results["QPSK"], "s-", color="C1", label="QPSK (sim)")
    ax.semilogy(snr_range, ber_qpsk_th, "--", color="C1", label="QPSK (theory)")
    ax.set(title="OFDM BER vs SNR (AWGN)",
           xlabel="SNR [dB]  (time-domain, per activity sheet)",
           ylabel="Bit Error Rate", ylim=(1e-5, 1))
    ax.grid(True, which="both"); ax.legend()
    _finish(fig, "task4_ber_vs_snr")

    # Print a message indicating that the simulated BER decreases as the SNR increases,
    # and display the simulated BER values for BPSK and QPSK
    # at specific SNR points (every 4 dB).
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
