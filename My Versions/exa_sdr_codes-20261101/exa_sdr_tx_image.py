# ============== PLUTO TRANSMITTER SCRIPT (exa_sdr_tx_image.py) ==============
# Builds one OFDM packet from an image (cat.jpeg) and transmits it
# continuously with an ADALM-Pluto SDR. Receive it with exa_sdr_rx_image.py.

# PlutoSDR in Python: https://pysdr.org/content/pluto.html
# ##########################################################################
# >>> WINDOWS-ONLY BLOCK - BEGIN >>>
#   WINDOWS : KEEP this block unchanged. It tells Python where to find the
#             libiio DLL and the C runtime (ucrtbase.dll), which Windows
#             cannot find on its own. Without it "import iio" fails.
#   macOS   : DELETE everything from this BEGIN line down to the END line
#             below. On a Mac these Windows paths break "import iio".
# ##########################################################################
# import ctypes.util
#
# original_find_library = ctypes.util.find_library
#
# def my_find_library(name):
#     if name in ("iio", "libiio"):
#         return r"C:\Windows\System32\libiio.dll"
#     if name == "c":
#         return "ucrtbase.dll"
#     return original_find_library(name)
#
# ctypes.util.find_library = my_find_library
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

# =======================================================================
#  1. TRANSMITTER PARAMETERS
# =======================================================================
#  This section defines all key parameters for the OFDM signal and the SDR.
#  These parameters must match the receiver's configuration.

# -- OFDM and Modulation --
mod_method = "BPSK"      # Modulation scheme. BPSK is robust and good for initial hardware testing.
n_fft = 64               # FFT size, defining the number of subcarriers.
n_cpe = 32               # Cyclic Prefix length. This acts as a guard interval to prevent multipath interference.
pilot_spacing = 4        # Insert a pilot tone every 4 subcarriers for better channel tracking.
pilot_val = 1 + 1j       # The known complex value of the pilot symbols.

# -- SDR and Channel Parameters --
carrier_freq = 2.4e9         # Radio frequency in Hz to transmit on (e.g., 2.4 GHz Wi-Fi band).
subcarrier_spacing = 15e3    # Spacing between subcarriers in Hz (15 kHz is a 5G standard).
sampling_rate = n_fft * subcarrier_spacing  # The required baseband sample rate for the system.
transmit_gain = -10            # SDR transmitter gain in dB (-80 to 0). 0 provides maximum power.

# -- Helper Variables --
mod_methods = ["BPSK", "QPSK"]
# 'mod_order' is the number of bits per symbol (e.g., BPSK -> 1, QPSK -> 2).
if mod_method not in mod_methods:
    raise ValueError(f"Unsupported modulation '{mod_method}'")
mod_order = mod_methods.index(mod_method) + 1

# =======================================================================
#  2. DATA PREPARATION & MODULATION (Resized Image)
# =======================================================================
#  This section prepares the source data (an image) for transmission.

# -- Step 2.1: Load and Resize the Image --
im_full = np.array(Image.open("cat.jpeg"))  # Load the original high-resolution image.
# Resize the image to a smaller dimension. This ensures the resulting data
# packet is small enough to fit within the PlutoSDR's transmit buffer
# while still representing the entire image.
im = np.array(Image.fromarray(im_full).resize((70, 70), resample=Image.BICUBIC))  # 70x70 pixels x 3 colours (RGB) x 8 bits = 117,600 bits.

# -- Step 2.2: Convert Resized Image to Binary Stream --
# Unpack bits (MSB-first) into a single 0/1 stream
im_bin = np.unpackbits(im.astype(np.uint8).ravel(), bitorder="big").astype(np.uint8)

# -- Step 2.3: Binary Stream to Integer Symbols --
# Calculate and add padding bits to make the stream length a multiple of the modulation order.
sym_rem = (mod_order - (len(im_bin) % mod_order)) % mod_order
if sym_rem:
    im_bin_padded = np.concatenate([im_bin, np.zeros(sym_rem, dtype=np.uint8)])
else:
    im_bin_padded = im_bin

# Group bits into symbols. For BPSK, each group has 1 bit.
cons_data = im_bin_padded.reshape(-1, mod_order)

# Convert each binary group (e.g., '1') into its decimal equivalent (e.g., 1).
weights = (2 ** np.arange(mod_order - 1, -1, -1)).astype(np.uint32)
cons_sym_id = (cons_data * weights).sum(axis=1).astype(np.int32)

# -- Step 2.4: Symbol Modulation --
# Create a constellation map (symbol book) for the chosen modulation.
if mod_order == 1:
    symbol_book = np.exp(1j * 2 * np.pi * np.arange(2) / 2).astype(np.complex64)
elif mod_order == 2:
    symbol_book = np.exp(1j * (2 * np.pi * np.arange(4) / 4 + np.pi / 4)).astype(np.complex64)
else:
    raise ValueError("This transmitter script implements BPSK and QPSK only.")

# Map the decimal symbol IDs to complex constellation points.
# 'X' is the final stream of frequency-domain symbols to be transmitted.
X = symbol_book[cons_sym_id].astype(np.complex64)

# =======================================================================
#  3. OFDM SYMBOL GENERATION
# =======================================================================
#  This section constructs the complete time-domain OFDM signal packet.

# -- Preamble Generation (Schmidl & Cox) --
# The preamble is a known signal at the start of the packet used by the receiver for detection and synchronization.
L = n_fft // 2  # Length of each half of the preamble.
rng = np.random.default_rng(seed=0)  # Use a fixed seed to ensure the random preamble is repeatable on the receiver.
p_time_half = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex64)  # Create one half with random complex data.
# Normalize the preamble's power to match the average power of the data symbols.
p_time_half = p_time_half / np.std(p_time_half) * np.mean(np.abs(symbol_book))
# The full time-domain preamble consists of the sequence repeated twice.
preamble_ofdm = np.concatenate([p_time_half, p_time_half]).astype(np.complex64)
# Add a cyclic prefix to the preamble.
preamble_cpe = np.concatenate([preamble_ofdm[-n_cpe:], preamble_ofdm]).astype(np.complex64)

# -- Pilot and Data Insertion --
# Determine which subcarriers will carry known pilots and which will carry data.
pilot_locs = np.arange(0, n_fft, pilot_spacing, dtype=np.int32)
data_locs = np.setdiff1d(np.arange(n_fft, dtype=np.int32), pilot_locs)
num_data_carriers = len(data_locs)

# Pad the data symbol stream to fit perfectly into the available data subcarriers.
data_padding_len = (num_data_carriers - (len(X) % num_data_carriers)) % num_data_carriers
if data_padding_len:
    X_padded_data = np.concatenate([X, np.zeros(data_padding_len, dtype=np.complex64)])
else:
    X_padded_data = X

# Reshape the data into parallel blocks, one column per OFDM symbol.
X_data_blocks = X_padded_data.reshape(-1, num_data_carriers).T  # shape: [num_data_carriers, num_symbols]
num_ofdm_data_symbols = X_data_blocks.shape[1]

# Create the final frequency-domain frames by mapping data and pilots to their subcarriers.
X_blocks = np.zeros((n_fft, num_ofdm_data_symbols), dtype=np.complex64)
X_blocks[data_locs, :] = X_data_blocks
X_blocks[pilot_locs, :] = np.complex64(pilot_val)

# -- IFFT and CP Addition for Data Symbols --
# The Inverse Fast Fourier Transform (IFFT) is the core of OFDM, converting
# the parallel frequency-domain subcarriers into a single time-domain signal.
x = np.fft.ifft(X_blocks, n=n_fft, axis=0).astype(np.complex64)
# Add a cyclic prefix to each data symbol.
X_cpe = np.vstack([x[-n_cpe:, :], x]).astype(np.complex64)

# -- Construct the Final Transmit Signal Packet --
# The final signal 'x_s' is the preamble followed by all the data symbols.
x_s = np.concatenate([preamble_cpe, X_cpe.flatten(order="F")]).astype(np.complex64)

# -- Scale the Packet for the PlutoSDR DAC --
x_s =np.concatenate([preamble_cpe, X_cpe.flatten(order="F")]).astype(np.complex64)

# 1. Normalize to +/- 1.0 first
peak = np.max(np.abs(x_s))
if peak > 0:
    x_s = x_s / peak

# 2. Scale up for the PlutoSDR DAC (16-bit signed integers)
# We use 2^14 (16384) to be safe. Max is ~32767, but we leave headroom 
# to strictly avoid digital clipping/overflow wrapping.
x_s = x_s * (2**14)

# =======================================================================
#  4. REAL-TIME TRANSMISSION
# =======================================================================
#  This section configures the PlutoSDR and begins transmission.

# -- Configure the PlutoSDR Transmitter Object --
# Set the Pluto address in connect_pluto() below: "ip:192.168.2.1", or the
# usb:... address shown by `iio_info -S usb`.

def fix_cyclic_tx_block_size(dev):
    """
    Fixes two bugs in pyadi-iio's continuous (cyclic) transmission with
    libiio 1.0. Without this, only part of the packet is transmitted:
      1. the hardware buffer size is given in samples where bytes are
         expected, so the buffer holds only 1/4 of the packet;
      2. only half of the packet data is copied into the buffer, so the
         rest is transmitted as zeros.
    Do not remove.
    """
    import iio
    print(f"libiio version: {iio.version}")
    if iio.version[0] < 1:
        return
    original_init = dev._tx_init_channels

    def init_with_bytes():
        n_samples = dev._tx_buffer_size
        dev._tx_buffer_size = n_samples * 4  # int16 I + int16 Q per sample
        try:
            original_init()
        finally:
            dev._tx_buffer_size = n_samples

    original_push = dev._tx_buffer_push

    def push_as_bytes(data):
        original_push(data.view(np.uint8))  # len() == number of bytes

    dev._tx_init_channels = init_with_bytes
    dev._tx_buffer_push = push_as_bytes


def connect_pluto():
    """
    Connect to the Pluto. Set its address in the adi.Pluto("...") line below:
    "ip:192.168.2.1", or the usb:... address shown by `iio_info -S usb`.
    """
    try:
        print("Connecting to PlutoSDR via USB...")
        dev = adi.Pluto("ip:192.168.2.1")
        _ = dev.sample_rate  # sanity check
        print("Connected to PlutoSDR via USB")
        fix_cyclic_tx_block_size(dev)
        return dev
    except Exception as e:
        raise RuntimeError(
            "USB connection to PlutoSDR failed. "
            "Check the address in adi.Pluto(...) with `iio_info -S usb`."
        ) from e


txPluto = connect_pluto()
txPluto.tx_lo = int(carrier_freq)
txPluto.sample_rate = int(sampling_rate)
txPluto.tx_rf_bandwidth = int(sampling_rate)
txPluto.tx_hardwaregain_chan0 = float(transmit_gain)

# -- Transmit Continuously --
# With the cyclic buffer enabled, the Pluto repeats the packet in hardware
# and tx() returns immediately.
txPluto.tx_buffer_size = len(x_s)      # buffer holds exactly one packet
txPluto.tx_cyclic_buffer = True
txPluto.tx(x_s.astype(np.complex64))  

print("Transmission started. Transmitting continuously...")
print("Press Ctrl+C in this command window to stop the transmitter.")

# This infinite loop is crucial. It keeps the script alive.
# If the script ends, the 'txPluto' object would be destroyed, stopping the transmission.
try:
    while True:
        time.sleep(1)  # Pause for 1 second to prevent high CPU usage.
except KeyboardInterrupt:
    pass
finally:
    # Best-effort cleanup: disable cyclic buffer and destroy TX buffer
    try:
        txPluto.tx_destroy_buffer()
    except Exception:
        pass


