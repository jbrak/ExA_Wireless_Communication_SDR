"""
================================================================================
 ExA SDR-Based Wireless Experiments
================================================================================

This script builds one OFDM *packet* and transmits it continuously with an
ADALM-Pluto SDR.  The packet structure is:

    [ preamble ] [ data sym 0 ] [ data sym 1 ] ... [ data sym M-1 ]

  * preamble : a known OFDM symbol sent TWICE (two identical 64-sample halves).
               The receiver uses it for three things at once:
                 - packet detection   (autocorrelation of the two halves)
                 - frequency offset    (phase between the two halves)
                 - channel estimation  (FFT of one half / known reference)
  * data     : OFDM symbols carrying the payload as BPSK or QPSK, each with a
               cyclic prefix.  Four known pilot subcarriers per symbol let the
               receiver track residual phase.

The modulation, FFT size and cyclic-prefix idea are exactly those built up in
Part 1, so the simulation and the hardware system stay consistent.

Receive it with exa_sdr_rx_text.py. The CONFIG parameters below MUST match
the receiver.
================================================================================
"""

# ##########################################################################
# >>> WINDOWS-ONLY BLOCK - BEGIN >>>
#   WINDOWS : KEEP this block unchanged. It tells Python where to find the
#             libiio DLL and the C runtime (ucrtbase.dll), which Windows
#             cannot find on its own. Without it "import iio" fails.
#   macOS   : DELETE everything from this BEGIN line down to the END line
#             below. On a Mac these Windows paths break "import iio".
# ##########################################################################
import ctypes.util

original_find_library = ctypes.util.find_library

def my_find_library(name):
    if name in ("iio", "libiio"):
        return r"C:\Windows\System32\libiio.dll"
    if name == "c":
        return "ucrtbase.dll"
    return original_find_library(name)

ctypes.util.find_library = my_find_library
# ##########################################################################
# <<< WINDOWS-ONLY BLOCK - END <<<
#   macOS: delete up to and including this line.
#   BOTH Windows and macOS: KEEP the lines below (import iio / import adi).
# ##########################################################################

import iio
import adi

print("IIO and ADI working")


import time
import numpy as np
from PIL import Image

import ctypes
import platform
import os
import sys
# ------------------------------------------------------------------ #
#  CONFIG  (must match the receiver)                                 #
# ------------------------------------------------------------------ #
MOD_METHOD = "BPSK"          # "BPSK" or "QPSK"
N_FFT      = 64              # FFT size / number of subcarriers
CP_LEN     = 16              # cyclic prefix length 
PILOT_VAL  = 1 + 0j         # known pilot symbol value
PREAMBLE_SEED = 12345        # fixed seed so the RX can regenerate the preamble
MESSAGE = "HELLO OFDM FROM PLUTO 123"   # payload (set USE_RANDOM_BITS for noise)
USE_RANDOM_BITS = False      # True -> send a random bit sequence instead of text
N_RANDOM_BITS = 2000

# -- SDR / RF parameters --
CARRIER_FREQ       = 2.4e9               # RF centre frequency [Hz]
SUBCARRIER_SPACING = 15e3                # 15 kHz (a 5G numerology)
SAMPLING_RATE      = N_FFT * SUBCARRIER_SPACING   # baseband sample rate
TRANSMIT_GAIN      = -10                 # tx_hardwaregain [dB], range -89..0
PLUTO_URI          = "ip:192.168.2.1"         # or the usb:... address from `iio_info -S usb`

# ------------------------------------------------------------------ #
#  Subcarrier allocation (simplified WiFi-style 64-FFT map)          #
#    - DC (bin 0) nulled : avoids Pluto LO/DC leakage                #
#    - band edges nulled : bins 27..37 act as guard bands            #
#    - 52 used = 48 data + 4 pilots                                  #
# ------------------------------------------------------------------ #
_GUARD = set([0]) | set(range(27, 38))
ACTIVE   = np.array([k for k in range(N_FFT) if k not in _GUARD])
PILOTS   = np.array([7, 21, 43, 57])
DATA_LOCS = np.array([k for k in ACTIVE if k not in set(PILOTS.tolist())])
MOD_ORDER = 1 if MOD_METHOD == "BPSK" else 2     # bits per symbol


# ================================================================== #
#  Modulation                   #
# ================================================================== #
_INV_SQRT2 = 1 / np.sqrt(2)


def modulate(bits):
    """BPSK: 0->-1, 1->+1.  QPSK Gray: (1-2*b1 + j(1-2*b0))/sqrt(2)."""
    bits = bits.astype(np.int8)          # avoid uint8 wraparound in 1-2*b
    if MOD_METHOD == "BPSK":
        return (2 * bits - 1).astype(np.complex64)
    b = bits.reshape(-1, 2)
    i = 1 - 2 * b[:, 1]
    q = 1 - 2 * b[:, 0]
    return ((i + 1j * q) * _INV_SQRT2).astype(np.complex64)


# ================================================================== #
#  Payload preparation                                               #
# ================================================================== #
def build_payload_bits():
    """Return the bit stream: [32-bit length header][message bits].
    The header lets the receiver discard padding and recover the exact text."""
    if USE_RANDOM_BITS:
        rng = np.random.default_rng(seed=1)
        msg_bits = rng.integers(0, 2, N_RANDOM_BITS).astype(np.uint8)
    else:
        msg_bytes = np.frombuffer(MESSAGE.encode("utf-8"), dtype=np.uint8)
        msg_bits = np.unpackbits(msg_bytes, bitorder="big")
    header = np.unpackbits(np.array([len(msg_bits)], dtype=">u4").view(np.uint8),
                           bitorder="big")            # 32-bit big-endian length
    return np.concatenate([header, msg_bits]).astype(np.uint8)


# ================================================================== #
#  Frame construction                                                #
# ================================================================== #
def make_preamble():
    """One known OFDM symbol repeated twice (no extra CP needed: the first
    copy is the guard interval for the second)."""
    rng = np.random.default_rng(seed=PREAMBLE_SEED)
    P = np.zeros(N_FFT, dtype=complex)
    P[ACTIVE] = rng.choice([-1.0, 1.0], size=len(ACTIVE))   # known BPSK pattern
    p_time = np.fft.ifft(P)
    return np.concatenate([p_time, p_time]).astype(np.complex64)


def add_cp(symbol):
    return np.concatenate([symbol[-CP_LEN:], symbol])


def build_frame():
    """Assemble the full baseband packet (preamble + data symbols)."""
    payload_bits = build_payload_bits()

    # pad bits to a whole number of modulation symbols
    rem = (MOD_ORDER - len(payload_bits) % MOD_ORDER) % MOD_ORDER
    if rem:
        payload_bits = np.concatenate([payload_bits, np.zeros(rem, dtype=np.uint8)])
    syms = modulate(payload_bits)

    # pad symbols to fill a whole number of OFDM symbols
    rem = (len(DATA_LOCS) - len(syms) % len(DATA_LOCS)) % len(DATA_LOCS)
    if rem:
        syms = np.concatenate([syms, np.zeros(rem, dtype=np.complex64)])
    data_blocks = syms.reshape(-1, len(DATA_LOCS))      # [num_symbols, 48]
    n_sym = data_blocks.shape[0]

    parts = [make_preamble()]
    for row in data_blocks:
        X = np.zeros(N_FFT, dtype=complex)
        X[DATA_LOCS] = row
        X[PILOTS] = PILOT_VAL
        parts.append(add_cp(np.fft.ifft(X)))
    frame = np.concatenate(parts).astype(np.complex64)

    print(f"  Modulation        : {MOD_METHOD} ({MOD_ORDER} bit/symbol)")
    print(f"  Payload bits      : {len(payload_bits)}")
    print(f"  OFDM data symbols : {n_sym}")
    print(f"  Frame length      : {len(frame)} samples "
          f"({len(frame)/SAMPLING_RATE*1e3:.2f} ms at {SAMPLING_RATE/1e6:.3f} MHz)")
    return frame


def scale_for_dac(frame):
    """Normalise to +/-1 then scale to the Pluto's 16-bit DAC range with
    headroom (2**14) to avoid clipping."""
    peak = np.max(np.abs(frame))
    if peak > 0:
        frame = frame / peak
    return (frame * (2 ** 14)).astype(np.complex64)


# ================================================================== #
#  Hardware                          #
# ================================================================== #
def fix_cyclic_tx_block_size(dev):
    """Fixes two bugs in pyadi-iio's continuous (cyclic) transmission with
    libiio 1.0. Without this, only part of the packet is transmitted:
      1. the hardware buffer size is given in samples where bytes are
         expected, so the buffer holds only 1/4 of the packet;
      2. only half of the packet data is copied into the buffer, so the
         rest is transmitted as zeros.
    Do not remove."""
    import iio
    print(f"libiio version: {iio.version}")
    if iio.version[0] < 1:
        return
    original_init = dev._tx_init_channels

    def init_with_bytes():
        n_samples = dev._tx_buffer_size
        dev._tx_buffer_size = n_samples * 4      # int16 I + int16 Q per sample
        try:
            original_init()
        finally:
            dev._tx_buffer_size = n_samples

    original_push = dev._tx_buffer_push

    def push_as_bytes(data):
        original_push(data.view(np.uint8))      # len() == number of bytes

    dev._tx_init_channels = init_with_bytes
    dev._tx_buffer_push = push_as_bytes


def connect_pluto():
    import adi
    try:
        print(f"Connecting to PlutoSDR via {PLUTO_URI} ...")
        dev = adi.Pluto(PLUTO_URI)
        _ = dev.sample_rate          # sanity check
        print("Connected to PlutoSDR")
        fix_cyclic_tx_block_size(dev)
        return dev
    except Exception as e:
        raise RuntimeError(
            f"Connection to PlutoSDR failed. Check that PLUTO_URI ({PLUTO_URI}) matches `iio_info -S usb`."
        ) from e


def main():
    print("=== Task 8: OFDM transmitter ===")
    frame = build_frame()
    x_s = scale_for_dac(frame)

    txPluto = connect_pluto()
    txPluto.tx_lo = int(CARRIER_FREQ)
    txPluto.sample_rate = int(SAMPLING_RATE)
    txPluto.tx_rf_bandwidth = int(SAMPLING_RATE)
    txPluto.tx_hardwaregain_chan0 = float(TRANSMIT_GAIN)

    # cyclic buffer -> hardware loops the packet without CPU involvement
    txPluto.tx_buffer_size = len(x_s)
    txPluto.tx_cyclic_buffer = True
    txPluto.tx(x_s)

    print("Transmitting continuously. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            txPluto.tx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()