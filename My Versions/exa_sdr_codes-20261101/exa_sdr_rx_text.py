"""
================================================================================
 ExA SDR-Based Wireless Experiments
================================================================================

Receives the text transmitted by exa_sdr_tx_text.py.  It captures IQ samples,
finds the packet using the repeated preamble, then runs the same receiver chain
we built in simulation:

    detect (autocorrelation of the two preamble halves)
      -> coarse carrier-frequency-offset (CFO) estimate + correction
      -> remove cyclic prefix, FFT each symbol
      -> channel estimation from the preamble  +  one-tap equalisation
      -> per-symbol pilot phase tracking
      -> demodulate -> parse header -> recover text -> BER

ALL parameters in the CONFIG block must be identical to exa_sdr_tx_text.py
(except RECEIVE_GAIN and PLUTO_URI, which are receiver settings).
Run the transmitter first, then this script.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_FIGS = os.environ.get("SAVE_FIGS", "0") == "1"

# ------------------------------------------------------------------ #
#  CONFIG      #
# ------------------------------------------------------------------ #
MOD_METHOD = "BPSK"          # "BPSK" or "QPSK"
N_FFT      = 64
CP_LEN     = 16
PILOT_VAL  = 1 + 0j
PREAMBLE_SEED = 12345
MESSAGE = "HELLO OFDM FROM PLUTO 1234"
USE_RANDOM_BITS = False
N_RANDOM_BITS = 2000

CARRIER_FREQ       = 2.4e9
SUBCARRIER_SPACING = 15e3
SAMPLING_RATE      = N_FFT * SUBCARRIER_SPACING
RECEIVE_GAIN       = 10
PLUTO_URI          = "usb:1.5.5"   # address from `iio_info -S usb`, or "ip:192.168.2.1"
DETECT_THRESHOLD   = 0.6      # on the normalised detection metric (0..1)

# subcarrier allocation (identical to transmitter)
_GUARD = set([0]) | set(range(27, 38))
ACTIVE    = np.array([k for k in range(N_FFT) if k not in _GUARD])
PILOTS    = np.array([7, 21, 43, 57])
DATA_LOCS = np.array([k for k in ACTIVE if k not in set(PILOTS.tolist())])
MOD_ORDER = 1 if MOD_METHOD == "BPSK" else 2
_INV_SQRT2 = 1 / np.sqrt(2)


# ================================================================== #
#  Shared definitions
# ================================================================== #
def demodulate(symbols):
    s = np.asarray(symbols)
    if MOD_METHOD == "BPSK":
        return (s.real > 0).astype(np.uint8)
    b1 = (s.real < 0).astype(np.uint8)
    b0 = (s.imag < 0).astype(np.uint8)
    return np.column_stack((b0, b1)).reshape(-1)


def expected_payload_bits():
    """Regenerate the exact transmitted bit stream (header + payload) so we can
    measure BER.  Matches build_payload_bits() in exa_sdr_tx_text.py."""
    if USE_RANDOM_BITS:
        rng = np.random.default_rng(seed=1)
        msg_bits = rng.integers(0, 2, N_RANDOM_BITS).astype(np.uint8)
    else:
        msg_bytes = np.frombuffer(MESSAGE.encode("utf-8"), dtype=np.uint8)
        msg_bits = np.unpackbits(msg_bytes, bitorder="big")
    header = np.unpackbits(np.array([len(msg_bits)], dtype=">u4").view(np.uint8),
                           bitorder="big")
    return np.concatenate([header, msg_bits]).astype(np.uint8)


def preamble_freq_pattern():
    """The known frequency-domain preamble pattern P[k] (BPSK on active bins)."""
    rng = np.random.default_rng(seed=PREAMBLE_SEED)
    P = np.zeros(N_FFT, dtype=complex)
    P[ACTIVE] = rng.choice([-1.0, 1.0], size=len(ACTIVE))
    return P


def make_preamble():
    """Time-domain preamble: one known OFDM symbol repeated twice."""
    p_time = np.fft.ifft(preamble_freq_pattern())
    return np.concatenate([p_time, p_time])


def num_data_symbols():
    """How many OFDM data symbols the transmitter produced for this payload."""
    nbits = len(expected_payload_bits())
    nbits += (MOD_ORDER - nbits % MOD_ORDER) % MOD_ORDER
    nsyms = nbits // MOD_ORDER
    nsyms += (len(DATA_LOCS) - nsyms % len(DATA_LOCS)) % len(DATA_LOCS)
    return nsyms // len(DATA_LOCS)


N_SYM = num_data_symbols()
PACKET_LEN = 2 * N_FFT + N_SYM * (CP_LEN + N_FFT)     # preamble + data symbols


# ================================================================== #
#  Receiver DSP                                                      #
# ================================================================== #
def detection_metric(r, L=N_FFT):
    """Normalised autocorrelation of two L-spaced halves (Schmidl & Cox style).
    Peaks (~1) where the two identical preamble copies align."""
    g = np.conj(r[:-L]) * r[L:]                 # r*[k] r[k+L]
    win = np.ones(L)
    P = np.convolve(g, win, mode="valid")        # sliding sum over L
    energy = np.convolve(np.abs(r[L:]) ** 2, win, mode="valid")
    M = np.abs(P) ** 2 / (energy ** 2 + 1e-12)
    return M, P


def estimate_cfo(packet, L=N_FFT):
    """Coarse CFO (Hz) from the phase between the two preamble halves."""
    P = np.sum(np.conj(packet[:L]) * packet[L:2 * L])
    return np.angle(P) * SAMPLING_RATE / (2 * np.pi * L)


def decode_packet(packet):
    """Run the full chain on one extracted packet. Returns a results dict."""
    L = N_FFT
    P_ref = preamble_freq_pattern()

    # -- coarse CFO correction --
    cfo = estimate_cfo(packet, L)
    n = np.arange(len(packet))
    packet_c = packet * np.exp(-1j * 2 * np.pi * cfo * n / SAMPLING_RATE)

    # -- channel estimate from the 2nd preamble copy (1st copy is its guard) --
    Yp = np.fft.fft(packet_c[L:2 * L])
    H = np.ones(N_FFT, dtype=complex)
    H[ACTIVE] = Yp[ACTIVE] / P_ref[ACTIVE]

    # -- per data symbol: CP removal, FFT, equalise, pilot phase correction --
    bits, eq_syms, pilot_phases = [], [], []
    base = 2 * N_FFT
    for i in range(N_SYM):
        s0 = base + i * (CP_LEN + N_FFT) + CP_LEN
        block = packet_c[s0:s0 + N_FFT]
        Y = np.fft.fft(block)
        Xeq = np.zeros(N_FFT, dtype=complex)
        Xeq[ACTIVE] = Y[ACTIVE] / H[ACTIVE]              # one-tap equaliser
        phi = np.angle(np.sum(Xeq[PILOTS] * np.conj(PILOT_VAL)))   # residual phase
        Xeq *= np.exp(-1j * phi)
        pilot_phases.append(phi)
        data = Xeq[DATA_LOCS]
        eq_syms.append(data)
        bits.append(demodulate(data))

    bits = np.concatenate(bits)
    eq_syms = np.concatenate(eq_syms)

    # -- parse 32-bit length header, recover message --
    hdr = int(np.packbits(bits[:32])[None].view(">u4")[0, 0])
    msg = ""
    if 0 < hdr <= len(bits) - 32:
        mbits = bits[32:32 + hdr]
        try:
            msg = np.packbits(mbits).tobytes().decode("utf-8", "replace")
        except Exception:
            msg = "<decode error>"

    # -- BER against the known transmitted bits --
    ref = expected_payload_bits()
    m = min(len(ref), len(bits))
    ber = float(np.sum(ref[:m] != bits[:m]) / m)

    return dict(cfo=cfo, H=H, bits=bits, eq_syms=eq_syms, msg=msg,
                header=hdr, ber=ber, pilot_phases=np.array(pilot_phases),
                packet_c=packet_c)


def find_packet(buffer):
    """Locate a fully-contained packet in the capture buffer.
    Returns (start_index, metric_peak) or (None, peak)."""
    M, _ = detection_metric(buffer)
    valid = len(M) - PACKET_LEN          # need the whole packet after start
    if valid <= 0:
        return None, 0.0
    start = int(np.argmax(M[:valid]))
    return (start, float(M[start])) if M[start] > DETECT_THRESHOLD else (None, float(M[start]))


# ================================================================== #
#  Visualisation / analysis                                          #
# ================================================================== #
def run_visualization_and_analysis(buffer, start, res):
    print(f"  Estimated CFO     : {res['cfo']:+.1f} Hz "
          f"({res['cfo'] / SUBCARRIER_SPACING:+.3f} subcarriers)")
    print(f"  Header length     : {res['header']} bits")
    print(f"  Recovered message : {res['msg']!r}")
    print(f"  BER vs transmitted: {res['ber']:.3e}")

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    ax[0, 0].plot(np.abs(buffer))
    ax[0, 0].axvline(start, color="r", ls="--", label="packet start")
    ax[0, 0].set(title="Captured |IQ|", xlabel="sample", ylabel="magnitude")
    ax[0, 0].legend(); ax[0, 0].grid(True)

    M, _ = detection_metric(buffer)
    ax[0, 1].plot(M)
    ax[0, 1].axvline(start, color="r", ls="--")
    ax[0, 1].axhline(DETECT_THRESHOLD, color="g", ls=":", label="threshold")
    ax[0, 1].set(title="Detection metric", xlabel="sample", ylabel="M (0..1)")
    ax[0, 1].legend(); ax[0, 1].grid(True)

    blk = res["packet_c"][2 * N_FFT + CP_LEN:2 * N_FFT + CP_LEN + N_FFT]
    spec = np.fft.fftshift(np.abs(np.fft.fft(blk)))
    fbins = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=1 / SAMPLING_RATE)) / 1e3
    ax[0, 2].plot(fbins, spec)
    ax[0, 2].set(title="First data symbol spectrum",
                 xlabel="frequency [kHz]", ylabel="|X|")
    ax[0, 2].grid(True)

    ax[1, 0].plot(ACTIVE, np.abs(res["H"][ACTIVE]), ".-")
    ax[1, 0].set(title="Channel estimate |H[k]|",
                 xlabel="subcarrier", ylabel="|H|")
    ax[1, 0].grid(True)

    ax[1, 1].plot(ACTIVE, np.angle(res["H"][ACTIVE]), ".-")
    ax[1, 1].set(title="Channel estimate angle(H[k])",
                 xlabel="subcarrier", ylabel="phase [rad]")
    ax[1, 1].grid(True)

    s = res["eq_syms"]
    ax[1, 2].scatter(s.real, s.imag, s=8, alpha=0.4, color="C1")
    ax[1, 2].set(title=f"Equalised constellation\nBER={res['ber']:.1e}",
                 xlabel="I", ylabel="Q", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    ax[1, 2].axhline(0, color="k", lw=0.5); ax[1, 2].axvline(0, color="k", lw=0.5)
    ax[1, 2].grid(True); ax[1, 2].set_aspect("equal")

    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig("rx_analysis.png", dpi=130)
        print("  [saved] rx_analysis.png")
    else:
        plt.show()


# ================================================================== #
#  Hardware                                                          #
# ================================================================== #
def connect_pluto_rx():
    import adi
    try:
        print(f"Connecting to PlutoSDR via {PLUTO_URI} ...")
        dev = adi.Pluto(PLUTO_URI)
        _ = dev.sample_rate
        print("Connected to PlutoSDR")
        return dev
    except Exception as e:
        raise RuntimeError(
            f"Connection to PlutoSDR failed. Check that PLUTO_URI ({PLUTO_URI}) matches `iio_info -S usb`."
        ) from e


def main():
    print("=== Task 8/9: OFDM receiver ===")
    print(f"  Expecting {N_SYM} data symbols, packet length {PACKET_LEN} samples")

    rxPluto = connect_pluto_rx()
    rxPluto.rx_lo = int(CARRIER_FREQ)
    rxPluto.sample_rate = int(SAMPLING_RATE)
    rxPluto.rx_rf_bandwidth = int(SAMPLING_RATE)
    rxPluto.gain_control_mode_chan0 = "manual"
    rxPluto.rx_hardwaregain_chan0 = float(RECEIVE_GAIN)
    rxPluto.rx_buffer_size = int(PACKET_LEN * 5)

    print("Listening for OFDM packets...")
    try:
        while True:
            buffer = rxPluto.rx()
            if buffer is None or len(buffer) == 0:
                print("No signal received, trying again...")
                continue
            buffer = buffer.astype(np.complex64)

            start, peak = find_packet(buffer)
            if start is None:
                print(f"No valid packet (metric peak {peak:.2f}); listening...")
                continue

            print(f"Packet detected at sample {start} (metric {peak:.2f}). Processing...")
            packet = buffer[start:start + PACKET_LEN]
            res = decode_packet(packet)
            run_visualization_and_analysis(buffer, start, res)
            print("Processing complete. Stopping receiver.")
            break
    finally:
        try:
            rxPluto.rx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()