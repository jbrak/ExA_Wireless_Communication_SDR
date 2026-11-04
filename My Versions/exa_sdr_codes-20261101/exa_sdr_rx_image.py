# ============== PLUTO RECEIVER SCRIPT (exa_sdr_rx_image.py) ==============
# Receives the image transmitted by exa_sdr_tx_image.py with an ADALM-Pluto SDR.
#
# Receiver chain:
#   1. timing   : cross-correlate with the known preamble to find the packet
#   2. CFO      : estimate and correct the carrier frequency offset
#   3. channel  : estimate the channel from the preamble, equalize (one tap)
#   4. tracking : correct the residual phase of each symbol using the pilots
#   5. decode   : demodulate, rebuild the image, compute BER, SNR, throughput
#
# Run the transmitter first, then this script.

# PlutoSDR in Python: https://pysdr.org/content/pluto.html


import time
import numpy as np
from PIL import Image
from scipy import signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import adi





def connect_pluto_rx():
    """
    Connect to the Pluto. Set its address in the adi.Pluto("...") line below:
    the usb:... address shown by `iio_info -S usb`, or "ip:192.168.2.1".
    """
    try:
        print("Connecting to PlutoSDR via USB (usb:0.3.5)...")
        dev = adi.Pluto("usb:1.5.5")
        _ = dev.sample_rate  # sanity check
        print("Connected to PlutoSDR via USB")
        return dev
    except Exception as e:
        raise RuntimeError(
            "USB connection to PlutoSDR failed. "
            "Check the address in adi.Pluto(...) with `iio_info -S usb`."
        ) from e



def run_visualization_and_analysis(
    X_hat_blocks,
    x_s_cfo,
    im,
    symbol_book,
    mod_order,
    data_locs,
    data_padding_len,
    sym_rem,
    im_bin,
    n_fft,
    n_cpe,
    sampling_rate,
    num_ofdm_data_symbols,
):
    # --- Reconstruct Corrected Data (After) ---
    # This block takes the final corrected symbols and decodes them back to bits.
    X_hat_data_only = X_hat_blocks[data_locs, :]
    X_hat = X_hat_data_only.flatten(order="F")
    if data_padding_len > 0:
        X_hat = X_hat[: -data_padding_len]  # Remove padding.

    const_points = np.column_stack([symbol_book.real, symbol_book.imag])
    tree = cKDTree(const_points)
    rx_points = np.column_stack([X_hat.real, X_hat.imag])
    _, nn_idx = tree.query(rx_points, k=1)
    rec_syms = nn_idx.astype(np.int64)  

    bit_shifts = np.arange(mod_order - 1, -1, -1, dtype=np.int64)
    rec_im_bits = ((rec_syms[:, None] >> bit_shifts) & 1).astype(np.uint8).flatten()

    if sym_rem > 0:
        rec_im_bits = rec_im_bits[: -sym_rem]

    min_len = min(len(rec_im_bits), len(im_bin))
    ber_after = float(np.mean(rec_im_bits[:min_len] != im_bin[:min_len]))

    # Reconstruct the image from the corrected bits.
    rec_bytes = np.packbits(rec_im_bits, bitorder="big")
    rec_im_after = np.zeros_like(im, dtype=np.uint8)
    n_copy = min(rec_bytes.size, im.size)
    rec_im_after.flat[:n_copy] = rec_bytes.flat[:n_copy]

    # --- Reconstruct Uncorrected Data (Before) ---
    # This block decodes the raw, uncorrected signal to get a baseline for performance.
    num_symbols = len(x_s_cfo) // (n_fft + n_cpe)
    X_p_uncorrected = x_s_cfo[: num_symbols * (n_fft + n_cpe)].reshape((n_fft + n_cpe, num_symbols), order="F")
    X_p_data_uncorrected = X_p_uncorrected[n_cpe:, 1:]  # exclude preamble symbol
    X_hat_blocks_uncorrected = np.fft.fft(X_p_data_uncorrected, n=n_fft, axis=0)

    X_hat_uncorrected_data = X_hat_blocks_uncorrected[data_locs, :]
    X_hat_before = X_hat_uncorrected_data.flatten(order="F")
    if data_padding_len > 0:
        X_hat_before = X_hat_before[: -data_padding_len]

    rx_points_before = np.column_stack([X_hat_before.real, X_hat_before.imag])
    _, nn_idx_before = tree.query(rx_points_before, k=1)
    rec_syms_before = nn_idx_before.astype(np.int64)

    rec_im_bits_before = ((rec_syms_before[:, None] >> bit_shifts) & 1).astype(np.uint8).flatten()
    if sym_rem > 0:
        rec_im_bits_before = rec_im_bits_before[: -sym_rem]

    min_len_b = min(len(rec_im_bits_before), len(im_bin))
    ber_before = float(np.mean(rec_im_bits_before[:min_len_b] != im_bin[:min_len_b]))

    rec_bytes_before = np.packbits(rec_im_bits_before, bitorder="big")
    rec_im_before = np.zeros_like(im, dtype=np.uint8)
    n_copy_b = min(rec_bytes_before.size, im.size)
    rec_im_before.flat[:n_copy_b] = rec_bytes_before.flat[:n_copy_b]

    # --- Create Visualization Figure ---
    plt.figure(figsize=(12, 8))

    # --- Top Row: Constellations and Original Image ---
    plt.subplot(2, 3, 1)

    # Normalize the uncorrected symbols so they fit in the plot.
    scale_factor = np.mean(np.abs(X_hat_before))
    X_hat_before_norm = X_hat_before / scale_factor
    
    # Plot the normalized version
    plt.plot(X_hat_before_norm[0::50].real, X_hat_before_norm[0::50].imag, "x", markersize=4)
    
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Received Constellation (Before)", fontweight="bold")

    plt.subplot(2, 3, 2)
    plt.plot(X_hat[0::50].real, X_hat[0::50].imag, "x", markersize=4)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Received Constellation (After)", fontweight="bold")

    plt.subplot(2, 3, 3)
    plt.imshow(im)
    plt.title("Transmit Image", fontweight="bold")
    plt.axis("off")

    # --- Bottom Row: Recovered Images and Metrics ---
    plt.subplot(2, 3, 4)
    plt.imshow(rec_im_before)
    plt.title(f"Recovered Image (Before)\nBER: {ber_before:.4f}", fontweight="bold")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(rec_im_after)
    plt.title(f"Recovered Image (After)\nBER: {ber_after:.4f}", fontweight="bold")
    plt.axis("off")

    # --- Final Metrics Calculation ---

    # -- Calculate Throughput --
    num_bits_transmitted = len(im_bin)
    symbol_duration = (n_fft + n_cpe) / sampling_rate
    total_time = num_ofdm_data_symbols * symbol_duration
    throughput_before = (num_bits_transmitted * (1 - ber_before)) / total_time
    throughput_after = (num_bits_transmitted * (1 - ber_after)) / total_time

    # -- Estimate SNR from Corrected Constellation --
    # This estimates the Signal-to-Noise-and-Distortion Ratio (SNDR) by comparing
    # the final corrected symbols to their ideal, noise-free positions.
    # (Recreate X_ideal exactly like transmitter mapping for the resized image)
    im_bin_padded_ref = np.concatenate([im_bin, np.zeros(sym_rem, dtype=np.uint8)]) if sym_rem > 0 else im_bin
    cons_data_ref = im_bin_padded_ref.reshape(-1, mod_order)
    weights = (2 ** np.arange(mod_order - 1, -1, -1)).astype(np.uint32)
    cons_sym_id_ref = (cons_data_ref * weights).sum(axis=1).astype(np.int32)
    X_ideal = symbol_book[cons_sym_id_ref]

    # The error is the vector difference between the ideal and received points.
    n_ideal = min(len(X_ideal), len(X_hat))
    error_vector = X_ideal[:n_ideal] - X_hat[:n_ideal]
    noise_plus_dist_power = np.mean(np.abs(error_vector) ** 2)
    signal_power = np.mean(np.abs(X_ideal[:n_ideal]) ** 2)
    snr_estimated = 10 * np.log10(signal_power / (noise_plus_dist_power + 1e-12))

    # -- Display Final Performance Summary in Command Window and Plot --
    summary_text = [
        "--- Performance Summary ---",
        f"Estimated Channel SNR: {snr_estimated:.2f} dB",
        "-----------------------------------",
        "Metric             Before      After",
        "-----------------------------------",
        f"BER:               {ber_before:.4f}    {ber_after:.4f}",
        f"Throughput (kbps): {throughput_before/1e3:.2f}     {throughput_after/1e3:.2f}",
    ]

    print("\n")
    for line in summary_text:
        print(line)
    print("")

    plt.subplot(2, 3, 6)
    plt.axis("off")
    plt.text(0.05, 0.5, "\n".join(summary_text), va="center", family="monospace", fontsize=10)
    plt.title("Performance Metrics", fontweight="bold")

    plt.tight_layout()
    plt.show()


# ========================================================================
#  1. RECEIVER PARAMETERS (MUST MATCH TRANSMITTER)
# ========================================================================
#  This section defines all key parameters for the receiver. These MUST be
#  identical to the parameters in exa_sdr_tx_image.py for the receiver to
#  be able to decode the signal correctly.

# -- OFDM and Modulation --
mod_method = "BPSK"      # Modulation scheme (must match transmitter).
n_fft = 64               # FFT size (number of subcarriers).
n_cpe = 32               # Cyclic Prefix length (Guard Interval).
pilot_spacing = 4        # Spacing of pilot tones for channel tracking.
pilot_val = 1 + 1j       # The known complex value of the pilots.

# -- SDR and Channel Parameters --
carrier_freq = 915e6     # Carrier frequency (must be tuned to the same frequency as the transmitter).
subcarrier_spacing = 15e3  # Subcarrier spacing in Hz.
sampling_rate = n_fft * subcarrier_spacing  # The baseband sampling rate of the system.
receive_gain = 25        # Receiver gain in dB. This is a critical parameter to tune for the specific environment.
phase_slope_tracking = False  # Pilot phase correction per symbol:
                              #   False: one common phase (residual CFO)
                              #   True : common phase + phase slope across
                              #          subcarriers (sampling-clock drift)

# -- Helper Variables --
mod_methods = ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM"]
# 'mod_order' is the number of bits per symbol (e.g., BPSK -> 1).
if mod_method not in mod_methods:
    raise ValueError(f"Unsupported modulation '{mod_method}'")
mod_order = mod_methods.index(mod_method) + 1

# ========================================================================
#  2. GENERATE LOCAL COPIES (Resized Image)
# ========================================================================
#  The receiver needs to generate its own ideal copies of the transmitted
#  signal components. These are used for synchronization (finding the
#  preamble), calculating the Bit Error Rate (comparing against the original
#  data), and for plotting reference signals.

# -- Generate Local Preamble --
L = n_fft // 2  # Length of one half of the Schmidl & Cox preamble.
rng = np.random.default_rng(seed=0)  # Use a fixed seed to ensure the random preamble matches the transmitter's.
p_time_half = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex64)

# -- Generate Local Original Data from Resized Image --
im_full = np.array(Image.open("cat.jpeg"))  # Load the original high-resolution image.
im = np.array(Image.fromarray(im_full).resize((70, 70), resample=Image.BICUBIC))  # Resize to the EXACT same 70x70 pixels as the transmitter.

im_bin = np.unpackbits(im.astype(np.uint8).ravel(), bitorder="big").astype(np.uint8)
sym_rem = (mod_order - (len(im_bin) % mod_order)) % mod_order

# -- Generate Symbol Book and Finalize Preamble --
if mod_order == 1:
    symbol_book = np.exp(1j * 2 * np.pi * np.arange(2) / 2).astype(np.complex64)
elif mod_order == 2:
    symbol_book = np.exp(1j * (2 * np.pi * np.arange(4) / 4 + np.pi / 4)).astype(np.complex64)
else:
    raise ValueError("This receiver script implements BPSK and QPSK only.")

# Finalize the preamble using the symbol book's average power for normalization.
p_time_half = p_time_half / np.std(p_time_half) * np.mean(np.abs(symbol_book))
preamble_ofdm = np.concatenate([p_time_half, p_time_half]).astype(np.complex64)  # Time-domain preamble for correlation.
preamble_seq = np.fft.fft(preamble_ofdm, n_fft).astype(np.complex64)  # Frequency-domain preamble for channel estimation.
preamble_cpe_ref = np.concatenate([preamble_ofdm[-n_cpe:], preamble_ofdm]).astype(np.complex64)  # Preamble with CP for reference plotting.

# -- Calculate Packet Length and Generate Ideal Transmitted Signal --
# This logic MUST perfectly mirror the transmitter's logic to correctly
# calculate the total packet length for data extraction.
pilot_locs_1b = np.arange(1, n_fft + 1, pilot_spacing) 
data_locs_1b = np.setdiff1d(np.arange(1, n_fft + 1), pilot_locs_1b)
num_data_carriers = len(data_locs_1b)

num_bits = len(im_bin) + sym_rem
num_syms = num_bits // mod_order
data_padding_len = (num_data_carriers - (num_syms % num_data_carriers)) % num_data_carriers
num_ofdm_data_symbols = (num_syms + data_padding_len) // num_data_carriers
packet_length = (n_fft + n_cpe) * (1 + num_ofdm_data_symbols)

# Generate a local, perfect copy of the transmitted data symbols for reference plots.
im_bin_padded = np.concatenate([im_bin, np.zeros(sym_rem, dtype=np.uint8)]) if sym_rem > 0 else im_bin
cons_data = im_bin_padded.reshape(-1, mod_order)
weights = (2 ** np.arange(mod_order - 1, -1, -1)).astype(np.uint32)
cons_sym_id = (cons_data * weights).sum(axis=1).astype(np.int32)
X = symbol_book[cons_sym_id].astype(np.complex64)

X_padded_data = np.concatenate([X, np.zeros(data_padding_len, dtype=np.complex64)]) if data_padding_len > 0 else X
X_data_blocks_ref = X_padded_data.reshape(-1, num_data_carriers).T  # [num_data_carriers, num_symbols]
X_blocks_ref = np.zeros((n_fft, num_ofdm_data_symbols), dtype=np.complex64)

# Convert 1-based locs to 0-based indices for Python arrays
pilot_locs = pilot_locs_1b - 1
data_locs = data_locs_1b - 1

X_blocks_ref[data_locs, :] = X_data_blocks_ref
X_blocks_ref[pilot_locs, :] = np.complex64(pilot_val)
x_ref = np.fft.ifft(X_blocks_ref, n=n_fft, axis=0).astype(np.complex64)
X_cpe_ref = np.vstack([x_ref[-n_cpe:, :], x_ref]).astype(np.complex64)

# ========================================================================
#  3. REAL-TIME RECEPTION AND PROCESSING
# ========================================================================
# -- Configure the PlutoSDR Receiver --
rxPluto = connect_pluto_rx()
rxPluto.rx_lo = int(carrier_freq)
rxPluto.sample_rate = int(sampling_rate)
rxPluto.rx_rf_bandwidth = int(sampling_rate)
rxPluto.gain_control_mode_chan0 = "manual"
rxPluto.rx_hardwaregain_chan0 = float(receive_gain)

rxPluto.rx_buffer_size = int(packet_length * 5)

print("Listening for OFDM packets...")
threshold = 0.7

# --- Continuous Loop for Packet Detection ---
# This loop will continuously capture frames of data from the SDR until
# a valid packet is detected and processed successfully, at which point it breaks.
while True:
    # 1. Receive a frame of data from the SDR.
    received_signal = rxPluto.rx()
    if received_signal is None or len(received_signal) == 0:
        print("No signal received, trying again...")
        continue

    received_signal = received_signal.astype(np.complex64)

    # 2. Find the preamble using cross-correlation.
    # This slides the known preamble across the received signal to find the point of best match.
    correlation = signal.correlate(received_signal, preamble_ofdm, mode="full")
    correlation_abs = np.abs(correlation)

    peak_idx = int(np.argmax(correlation_abs))  # Find the location of the strongest match.
    peak_val = float(correlation_abs[peak_idx])

    lags = np.arange(-(len(preamble_ofdm) - 1), len(received_signal), dtype=np.int64)

    # 3. Check if the detected peak is strong enough to be a valid packet.
    # The threshold is based on the energy of the preamble to be robust.
    if peak_val > threshold * float(np.sum(np.abs(preamble_ofdm) ** 2)):
        print("Packet detected! Processing...")

        # 4. Extract the full packet from the large received buffer.
        packet_start_idx = int(lags[peak_idx])  # 0-based in Python

        # The preamble with its CP is three identical 32-sample copies, so the
        # correlation has two equal peaks L samples apart. The earlier one is the
        # true packet start; if argmax landed on the later one, step back by L.
        before = correlation_abs[peak_idx - L] if peak_idx >= L else 0.0
        after = correlation_abs[peak_idx + L] if peak_idx + L < len(correlation_abs) else 0.0
        if before > after:
            packet_start_idx -= L
        # Back off a few samples into the CP. This only adds a linear phase across
        # subcarriers, which the channel estimate absorbs, and guards against
        # starting late (which would cause inter-symbol interference).
        packet_start_idx -= 4
        print(f"  Timing: {'stepped back to earlier peak' if before > after else 'earlier peak picked'}")

        # Ensure the full packet is contained within the captured frame.
        if (packet_start_idx >= 0) and (packet_start_idx + packet_length <= len(received_signal)):
            # 'x_s_cfo' is the extracted, raw packet, distorted by the real-world channel.
            x_s_cfo = received_signal[packet_start_idx : packet_start_idx + packet_length].astype(np.complex64)

            # --- FULL RECEIVER CHAIN ---
            # This chain mirrors the simulation, but operates on real data.

            # -- Coarse CFO Estimation and Correction --
            received_preamble_td = x_s_cfo[: n_fft + n_cpe]
            received_preamble = received_preamble_td[n_cpe:]
            P_est = np.sum(np.conj(received_preamble[:L]) * received_preamble[L : 2 * L])
            cfo_est_normalized = np.angle(P_est) / np.pi

            # Apply a "counter-rotation" to the whole packet to correct the bulk CFO.
            if np.abs(cfo_est_normalized) > 0.001:
                t_corr = np.arange(len(x_s_cfo), dtype=np.float64)
                cfo_corr_exp = np.exp(-1j * 2 * np.pi * cfo_est_normalized / n_fft * t_corr).astype(np.complex64)
                x_s_corrected = x_s_cfo * cfo_corr_exp
            else:
                x_s_corrected = x_s_cfo

            # -- Reshape, CP Removal, and FFT --
            num_ofdm_symbols_total = len(x_s_corrected) // (n_fft + n_cpe)
            X_p = x_s_corrected[: num_ofdm_symbols_total * (n_fft + n_cpe)].reshape(
                (n_fft + n_cpe, num_ofdm_symbols_total), order="F"
            )
            X_p_data = X_p[n_cpe:, 1:]
            X_hat_blocks = np.fft.fft(X_p_data, n=n_fft, axis=0)  # Convert data symbols to frequency domain.

            # -- Channel Estimation and Equalization --
            # Use the corrected preamble to estimate the channel's frequency response.
            received_preamble_freq = np.fft.fft(X_p[n_cpe:, 0], n=n_fft)
            # The preamble repeats every 32 samples, so it only has energy on the
            # EVEN subcarriers (preamble_seq is exactly 0 on odd ones). Estimate the
            # channel on even subcarriers and interpolate to the odd ones.
            G = np.zeros(n_fft, dtype=np.complex64)
            G[0::2] = received_preamble_freq[0::2] / preamble_seq[0::2]
            G[1::2] = (G[0::2] + np.roll(G[0::2], -1)) / 2

            # Divide the data symbols by the channel response to equalize them.
            X_hat_blocks = X_hat_blocks / G[:, None]

            if phase_slope_tracking:
                # -- Fine Phase Tracking using Pilots (common phase + slope) --
                # Per symbol, fit a common phase (residual CFO) and a phase slope across
                # subcarriers (sampling-clock drift). The DC pilot (bin 0, LO leakage)
                # and the band-edge pilot (bin 32) are unreliable, so they are skipped.
                k_signed = np.where(np.arange(n_fft) < n_fft // 2, np.arange(n_fft), np.arange(n_fft) - n_fft)
                track = np.array([p for p in pilot_locs if p not in (0, n_fft // 2)])
                track = track[np.argsort(k_signed[track])]  # order by frequency
                neighbours = np.diff(k_signed[track]) == pilot_spacing
                for i in range(X_hat_blocks.shape[1]):
                    p = X_hat_blocks[track, i] * np.conj(np.complex64(pilot_val))
                    slope = np.angle(np.sum((p[1:] * np.conj(p[:-1]))[neighbours])) / pilot_spacing
                    phase = np.angle(np.sum(p * np.exp(-1j * slope * k_signed[track])))
                    X_hat_blocks[:, i] *= np.exp(-1j * (phase + slope * k_signed)).astype(np.complex64)
            else:
                # -- Fine Phase Correction using Pilots (common phase) --
                # Corrects residual CFO and phase noise on a symbol-by-symbol basis.
                for i in range(X_hat_blocks.shape[1]):
                    received_pilots = X_hat_blocks[pilot_locs, i]
                    phase_error = np.angle(np.mean(received_pilots * np.conj(np.complex64(pilot_val))))
                    X_hat_blocks[:, i] = X_hat_blocks[:, i] * np.exp(-1j * phase_error).astype(np.complex64)

            print(f"  CFO estimate: {cfo_est_normalized * sampling_rate / n_fft:+.0f} Hz")

            # -- Call the visualization and analysis function --
            run_visualization_and_analysis(
                X_hat_blocks,
                x_s_cfo,
                im,
                symbol_book,
                mod_order,
                data_locs,
                data_padding_len,
                sym_rem,
                im_bin,
                n_fft,
                n_cpe,
                sampling_rate,
                num_ofdm_data_symbols,
            )

            # -- Break the loop after successful processing --
            print("Processing complete. Stopping receiver.")
            break
        else:
            print("Detected preamble, but packet not fully contained in this frame. Listening again...")
    else:
        print("No valid packet detected in this frame, listening again...")

# Release the SDR object to free the hardware for other applications.
try:
    rxPluto.rx_destroy_buffer()
except Exception:
    pass
del rxPluto
