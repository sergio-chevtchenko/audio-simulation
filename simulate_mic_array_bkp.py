import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram

def load_sound_source(filepath, expected_fs=None):
    """
    Load a sound source from a WAV file.

    Parameters:
        filepath (str): Path to the WAV file.
        expected_fs (int): Expected sampling frequency. If None, doesn't check the fs.

    Returns:
        np.array: Audio signal extracted from the WAV file.
    """
    fs, signal = wavfile.read(filepath)

    if expected_fs is not None and fs != expected_fs:
        raise ValueError(f"Sampling frequency of file {filepath} is {fs}, expected {expected_fs}")

    # Ensure the signal is mono
    if len(signal.shape) == 2:
        signal = signal[:, 0]

    return fs, signal

# [Global Parameters]
speed_of_sound = 343  # m/s
fs = 44100  # Sampling frequency

# Create anechoic room (free-field simulation)
room = pra.AnechoicRoom(fs=fs)

# Expected fs helps to check the consistency across the loaded signals.
fs_source, source_signal_1 = load_sound_source('frequency_sweep_triple_up.wav', expected_fs=fs)
_, source_signal_2 = load_sound_source('frequency_sweep_triple_down.wav', expected_fs=fs)

source_signals = [source_signal_1, source_signal_2]
#source_signals = [source_signal_1]

source_positions = [[0, 10, 0], [10, 10, 0]]

for sig, pos in zip(source_signals, source_positions):
    room.add_source(pos, signal=sig)

# Added mic_positions as a parameter so that it can be easily modified.
mic_positions = [[2.5, 1, 0], [7.5, 1, 0], [5, 3, 0]]
mic_array = pra.MicrophoneArray(np.array(mic_positions).T, room.fs)
room.add_microphone_array(mic_array)

source_positions = np.array(source_positions).astype(float)
mic_positions = np.array(mic_positions).astype(float)

def calculate_travel_times(source_positions, mic_positions, speed_of_sound):
    """
    Calculate the time it takes for the signal from each source to reach each microphone.

    Parameters:
        source_positions (np.array): Positions of the sound sources.
        mic_positions (np.array): Positions of the microphones.
        speed_of_sound (float): Speed of sound in the room, typically around 343 m/s.

    Returns:
        np.array: Matrix (num_sources x num_mics) of travel times from each source to each microphone.
    """
    num_sources = source_positions.shape[0]
    num_mics = mic_positions.shape[0]
    times = np.zeros((num_sources, num_mics))

    for i, src_pos in enumerate(source_positions):
        for j, mic_pos in enumerate(mic_positions):
            distance = np.sqrt(np.sum((src_pos - mic_pos)**2))
            times[i, j] = distance / speed_of_sound

    return times

# Calculating travel times
times = calculate_travel_times(source_positions, mic_positions, speed_of_sound)

# Displaying times relative to the first source-microphone pair
first_time = times[0, 0]

print("Times relative to time from source 1 to microphone 1:")
# More explicit variable names for clarity.
for mic_idx, mic_pos in enumerate(mic_positions):
    for src_idx, src_pos in enumerate(source_positions):
        relative_time = times[src_idx, mic_idx] - first_time
        print(f"Time from source {src_idx+1} to microphone {mic_idx+1}: "
              f"{relative_time*1000:.2f} ms ; {relative_time*fs:.0f} samples")
    print("\n")


def visualize_room(source_positions, mic_positions):
    """
    Create a 2D plot to visualize the room setup including source and microphone positions.

    Parameters:
        source_positions (list of lists or np.array): Positions of sources in the room.
        mic_positions (list of lists or np.array): Positions of microphones in the room.
    """
    plt.figure(figsize=(10, 8))

    # Plot sources
    source_positions = np.array(source_positions)
    plt.scatter(source_positions[:, 0], source_positions[:, 1], color='r', s=100, marker='o', label='Sources')

    # Annotating source positions
    for i, pos in enumerate(source_positions):
        plt.annotate(f"Src {i+1}", (pos[0], pos[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='r')

    # Plot microphones
    mic_positions = np.array(mic_positions)
    plt.scatter(mic_positions[:, 0], mic_positions[:, 1], color='b', s=100, marker='x', label='Microphones')

    # Annotating microphone positions
    for i, pos in enumerate(mic_positions):
        plt.annotate(f"Mic {i+1}", (pos[0], pos[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='b')

    # Set labels and title
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("2D Visualization of Room Setup")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Equal scaling

    # Show the plot
    # plt.show()

# Call the function to visualize the room setup
visualize_room(source_positions, mic_positions)

# Compute the RIR (Room Impulse Response)
room.compute_rir()

# Simulate sound propagation
room.simulate()

# Extract the simulated recordings (includes the sound signal, room effects, and any additional noise)
recordings = mic_array.signals

def sliding_window_fft(signals, window_size, step_size, fs, window_func=None,
                       live_plot=True, num_bins_freq=100, num_bins_time=100) -> np.array:
    """
    Compute, bin, and optionally plot the FFT for each window of the signals
    for all microphones. Binning is performed in both the frequency and time domain.

    Parameters:
        signals (np.array): The signals from all microphones. Shape: (num_mics, signal_length)
        window_size (int): Size of the window for the FFT.
        step_size (int): Step size between windows.
        fs (int): Sampling frequency.
        window_func (str, optional): Window function ('hamming', 'hanning'). Defaults to None.
        live_plot (bool): Whether to dynamically plot the FFT for each window. Defaults to True.
        num_bins_freq (int): Number of frequency bins. Defaults to 100.
        num_bins_time (int): Number of time bins. Defaults to 100.

    Returns:
        np.array: 3D array (num_mics x num_bins_freq x num_bins_time) containing
                  binned spectrogram data.
    """
    num_mics, signal_length = signals.shape
    num_windows = (signal_length - window_size) // step_size + 1

    if window_func == 'hamming':
        win_func = np.hamming(window_size)
    elif window_func == 'hanning':
        win_func = np.hanning(window_size)
    else:
        win_func = np.ones(window_size)

    spectrograms = np.zeros((num_mics, num_bins_freq, num_windows))

    if live_plot:
        fig_live = plt.figure(figsize=(12, 8))

    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + window_size

        if live_plot:
            fig_live.clf()

        for mic_idx in range(num_mics):
            window = signals[mic_idx, window_start:window_end] * win_func
            fft_vals = np.fft.fft(window)
            freqs = np.fft.fftfreq(window_size, 1/fs)

            bin_indices = np.linspace(0, window_size//2, num_bins_freq, endpoint=False, dtype=int)
            binned_fft_vals = np.array([np.mean(np.abs(fft_vals[j: j+2])) for j in bin_indices])
            spectrograms[mic_idx, :, i] = binned_fft_vals

            if live_plot:
                plt.subplot(num_mics, 1, mic_idx+1)
                plt.plot(freqs[bin_indices], binned_fft_vals, label=f"Mic {mic_idx+1}")
                plt.title(f"FFT of Window {i+1} of {num_windows+1}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.pause(0.01)

    if live_plot:
        plt.show()

    num_bins_time = min(num_bins_time, num_windows)
    bin_size_time = num_windows // num_bins_time

    binned_spectrograms_time = np.zeros((num_mics, num_bins_freq, num_bins_time))

    for mic_idx in range(num_mics):
        for i in range(num_bins_time):
            start_window = i * bin_size_time
            end_window = (i + 1) * bin_size_time
            binned_spectrograms_time[mic_idx, :, i] = np.mean(spectrograms[mic_idx, :, start_window:end_window], axis=1)

        plt.figure(figsize=(12, 8))
        plt.imshow(np.log(binned_spectrograms_time[mic_idx] + 1e-10), aspect='auto', cmap='inferno',
                   origin='lower', extent=[0, signal_length/fs, 0, fs/2], interpolation='none')
        plt.title(f"Double Binned Spectrogram - Mic {mic_idx+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Log amplitude')
        plt.tight_layout()

    return binned_spectrograms_time



# Parameters
window_size = 200  # Window size for FFT
step_size = 2  # Step size between windows
num_bins_freq = 100
num_bins_time = 200

# Perform sliding window FFT on each microphone recording.
# Assuming 'recordings' is a 2D array: num_mics x signal_length.

print("Performing sliding window FFT for all microphones")
spectrograms = sliding_window_fft(recordings, window_size, step_size, fs, window_func='hanning', live_plot=False, num_bins_freq=num_bins_freq, num_bins_time=num_bins_time)

# def find_time_delays(spectrograms, start_col, n, fs, debug_plots=True):
#     num_mics, num_freq_bins, num_time_bins = spectrograms.shape
#     assert num_mics >= 3, "At least three microphone spectrograms should be provided."
#
#     sub_spectrogram = spectrograms[0, :, start_col:start_col+n]
#
#     delays = np.zeros((num_freq_bins, num_mics-1))
#
#     if debug_plots:
#         fig_debug_plots = plt.figure(figsize=(14, 8))
#
#     for freq_idx in range(num_freq_bins):
#         target_row = sub_spectrogram[freq_idx, :]
#
#         for mic_idx in range(1, num_mics):
#             test_row = spectrograms[mic_idx, freq_idx, :]  # Using full row
#
#             corr = np.correlate(target_row, test_row, mode='full')
#             delay_idx = np.argmax(corr)
#             best_delay = delay_idx - (n - 1)
#             delays[freq_idx, mic_idx-1] = best_delay / fs
#
#
#             if debug_plots:
#                 fig_debug_plots.clf()
#
#                 # Plotting the entire spectrogram along with the target_row and test_row
#                 plt.subplot(3, 1, 1)
#                 plt.imshow(spectrograms[mic_idx, :, :], aspect='auto', origin='lower', cmap='jet',
#                         extent=[0, num_time_bins, 0, num_freq_bins])
#                 plt.colorbar(label='Amplitude')
#                 plt.title(f"Full Spectrogram - Mic: {mic_idx}, Freq Idx: {freq_idx}")
#                 plt.xlabel('Time Bin Index')
#                 plt.ylabel('Frequency Bin Index')
#                 plt.axhline(y=freq_idx, color='white', linestyle='--', label='Current Freq Bin')
#                 plt.axvline(x=start_col, color='white', linestyle='-', label='Start Col')
#                 plt.axvline(x=start_col+n, color='white', linestyle='-', label='End Col')
#                 plt.legend()
#
#                 # Plotting the target_row and test_row
#                 plt.subplot(3, 1, 2)
#                 plt.title(f"Target and Test Rows - Freq Idx: {freq_idx}, Mic: {mic_idx}")
#                 plt.plot(target_row, label='Target Row', linestyle='dashed')
#                 plt.plot(test_row, label='Test Row', linestyle='dotted')
#                 plt.legend()
#                 plt.xlabel('Time Bin Index')
#                 plt.ylabel('Amplitude')
#
#                 # Plotting the cross-correlation
#                 plt.subplot(3, 1, 3)
#                 plt.plot(corr, label='Cross-correlation', color='red')
#                 plt.axvline(x=delay_idx, color='green', linestyle='--', label='Max Correlation Index')
#                 plt.legend()
#                 plt.xlabel('Lag')
#                 plt.ylabel('Correlation')
#
#                 plt.tight_layout()
#
#                 plt.pause(0.01)
#
#     if debug_plots:
#         plt.show()
#
#     return delays
#
#
# def estimate_source_positions(delays, mic_positions, c=343):
#     """
#     Estimate the 3D position of sound sources based on time delays at each microphone.
#
#     Parameters:
#         delays (np.array): 2D array containing time delays (in seconds) for each microphone
#                            relative to the first one for each frequency bin.
#                            Shape: (num_freq_bins, num_mics-1)
#         mic_positions (np.array): 2D array containing the 3D positions of the microphones.
#                                   Shape: (num_mics, 3)
#         c (float): Speed of sound in medium (m/s). Defaults to 343 m/s in air.
#
#     Returns:
#         np.array: Estimated 3D positions of the sound sources for each frequency bin.
#                   Shape: (num_freq_bins, 3)
#     """
#     num_freq_bins, num_mics_minus_one = delays.shape
#     num_mics = num_mics_minus_one + 1
#     assert mic_positions.shape[0] == num_mics, "Mismatch in the number of microphones."
#
#     # Storage for estimated source positions for each frequency bin
#     estimated_positions = np.zeros((num_freq_bins, 3))
#
#     # Iterating through each frequency bin
#     for freq_idx in range(num_freq_bins):
#         # Solving for source position can be complex and might use optimization or root-finding
#         # algorithms based on the acoustic model and microphone setup.
#         # A basic direct solution approach is taken here for simplification.
#
#         # Example: Assume we have at least 3 mics and will use a basic triangulation method
#         # which might not be accurate for actual applications with noise and reflections.
#         if num_mics >= 3:
#             mic_1 = mic_positions[0]
#             mic_2 = mic_positions[1]
#             mic_3 = mic_positions[2]
#
#             # Example: Assuming the source is equidistant from mics 2 and 3, and we use the
#             # delay between mic 1 and mic 2 to estimate the position.
#             # Note: This is a simplified scenario, not suitable for robust localization.
#             t_12 = delays[freq_idx, 0]
#             # Finding mid-point between mic 2 and mic 3
#             mid_23 = (mic_2 + mic_3) / 2
#             # The direction vector from mic 1 to the mid-point between mic 2 and mic 3
#             direction_1_mid_23 = mid_23 - mic_1
#             direction_1_mid_23 = direction_1_mid_23 / np.linalg.norm(direction_1_mid_23)
#             # Assuming constant speed of sound, calculate the distance between mic 1
#             # and the source.
#             d_1 = c * t_12
#             # Estimate the source position
#             source_pos = mic_1 + direction_1_mid_23 * d_1
#
#             estimated_positions[freq_idx, :] = source_pos
#
#         else:
#             raise ValueError("At least 3 microphones are required for localization")
#
#     return estimated_positions
#
# def estimate_positions_over_time(spectrograms, n, fs, mic_positions, c=343):
#     """
#     Estimate the 3D positions and their corresponding times of sound sources by
#     evaluating each n-column slice of the spectrograms.
#
#     Parameters:
#         spectrograms (np.array): ...
#         n (int): ...
#         fs (int): ...
#         mic_positions (np.array): ...
#         c (float): ...
#
#     Returns:
#         np.array: Concatenated estimated 3D positions for each n-column slice.
#                   Shape: (num_slices, num_freq_bins, 3)
#         np.array: Corresponding times for each estimated position.
#                   Shape: (num_slices,)
#     """
#     _, _, num_time_bins = spectrograms.shape
#
#     # Assuming step size = n (no overlap), modify as needed
#     num_slices = num_time_bins // n
#
#     all_estimated_positions = []
#     estimated_times = []
#
#     for slice_idx in range(num_slices):
#         start_col = slice_idx * n
#         central_time = (start_col + (n/2)) / fs  # Center time of the slice
#
#         # Find delays and estimate positions for this slice
#         delays = find_time_delays(spectrograms, start_col, n, fs)
#
#         estimated_positions = estimate_source_positions(delays, mic_positions, c)
#
#         # Store the results
#         all_estimated_positions.append(estimated_positions)
#         estimated_times.append(central_time)
#
#     return np.array(all_estimated_positions), np.array(estimated_times)
#
#
# all_estimated_positions, estimated_times = estimate_positions_over_time(
#     spectrograms=spectrograms,
#     n=5,  # example window width
#     fs=fs,
#     mic_positions=mic_positions,
#     c=speed_of_sound
# )
#
# #print(all_estimated_positions)
# #print(estimated_times)
#
# def plot_positions_over_time(all_estimated_positions, estimated_times):
#     """
#     Plot the estimated positions over time in 3D space for all frequency bins.
#
#     Parameters:
#         all_estimated_positions (np.array): Estimated 3D positions for each n-column slice.
#                                             Shape: (num_slices, num_freq_bins, 3)
#         estimated_times (np.array): Corresponding times for each estimated position.
#                                     Shape: (num_slices,)
#     """
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     num_slices, num_freq_bins, _ = all_estimated_positions.shape
#
#     # Creating a colormap
#     cm = plt.get_cmap('viridis')
#     colors = [cm(1.*i/num_freq_bins) for i in range(num_freq_bins)]
#
#     for freq_bin_idx in range(num_freq_bins):
#         # Extracting x, y, and z coordinates. Assuming z is fixed, modify if needed.
#         x_coords = all_estimated_positions[:, freq_bin_idx, 0]
#         y_coords = all_estimated_positions[:, freq_bin_idx, 1]
#         z_coords = estimated_times  # Here z axis represents time
#
#         # Creating a scatter plot for each frequency bin
#         ax.scatter(x_coords, y_coords, z_coords, label=f'Freq Bin {freq_bin_idx}', color=colors[freq_bin_idx])
#
#         # Optionally creating lines to show evolution over time
#         ax.plot(x_coords, y_coords, z_coords, color=colors[freq_bin_idx], alpha=0.5)
#
#     # Labeling axes
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_zlabel('Time (s)')
#
#     # Title, legend, and grid
#     ax.set_title('Position Estimations Over Time')
#     ax.legend()
#     ax.grid(True)
#
# plot_positions_over_time(all_estimated_positions, estimated_times)

plt.show()
