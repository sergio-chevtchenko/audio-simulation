import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.signal import spectrogram
from scipy.optimize import fsolve
import seaborn as sns

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

source_positions = [[0, 20, 0], [20, 20, 0]]

for sig, pos in zip(source_signals, source_positions):
    room.add_source(pos, signal=sig)

# Added mic_positions as a parameter so that it can be easily modified.
mic_positions = [[7, 0, 0], [10, 3, 0], [13, 0, 0]]
mic_array = pra.MicrophoneArray(np.array(mic_positions).T, room.fs)
room.add_microphone_array(mic_array)

source_positions = np.array(source_positions).astype(float)
mic_positions = np.array(mic_positions).astype(float)

# Parameters
fft_window_size = 500  # Window size for FFT
step_size = 1  # Step size between windows
num_bins_freq = 200
num_bins_time = 500

# Parameters for sliding window
window_size_kHz_ms=(2, 10)  # Size of the window for sliding window analysis (kHz, ms)
search_amp_ms = 50  # Search amplitude in milliseconds for finding similar patches

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

    return binned_spectrograms_time


def plot_binned_spectrograms(binned_spectrograms_time, fs, signal_length, num_mics):
    """
    Plot the binned spectrograms for all microphones.

    Parameters:
        binned_spectrograms_time (np.array): The binned spectrogram data.
        fs (int): Sampling frequency.
        signal_length (int): Length of the signal.
        num_mics (int): Number of microphones.
    """

    duration = signal_length/fs

    for mic_idx in range(num_mics):
        plt.figure(figsize=(12, 8))
        plt.imshow(binned_spectrograms_time[mic_idx], aspect='auto', cmap='inferno',
                   origin='lower', extent=[0, duration, 0, fs/2], interpolation='none')
        plt.title(f"Double Binned Spectrogram - Mic {mic_idx+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Amplitude')
        plt.tight_layout()


def plot_spectrogram_statistics(binned_spectrograms_time, fs):
    """
    Plot statistical information about the binned spectrograms.

    Parameters:
        binned_spectrograms_time (np.array): The binned spectrogram data.
        fs (int): Sampling frequency.
    """

    num_mics, num_bins_freq, _ = binned_spectrograms_time.shape

    for mic_idx in range(num_mics):
        mean_spectrum = np.mean(binned_spectrograms_time[mic_idx], axis=1)
        median_spectrum = np.median(binned_spectrograms_time[mic_idx], axis=1)
        std_spectrum = np.std(binned_spectrograms_time[mic_idx], axis=1)

        freq_bins = np.linspace(0, fs/2, num_bins_freq)

        plt.figure(figsize=(12, 8))

        plt.plot(freq_bins, mean_spectrum, label='Mean')
        plt.plot(freq_bins, median_spectrum, label='Median')
        plt.fill_between(freq_bins, mean_spectrum-std_spectrum, mean_spectrum+std_spectrum, alpha=0.2, label='Std Dev')

        plt.title(f"Spectrogram Statistics - Mic {mic_idx+1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()

def process_spectrograms(spectrograms, threshold=1000):
    """
    Process the spectrogram data: zeroing values below a threshold, taking the log and normalizing.

    Parameters:
        spectrograms (np.array): The binned spectrogram data.
        threshold (float): Absolute value threshold below which values are set to zero.

    Returns:
        np.array: Processed spectrogram data.
    """

    # Set values with absolute value below threshold to zero
    spectrograms[np.abs(spectrograms) < threshold] = 0

    # Convert to logarithm
    spectrograms = np.log(spectrograms + 1e-10)  # Adding a small constant to avoid log(0)

    # Normalize to [0, 1]
    spectrograms -= np.min(spectrograms)
    spectrograms /= np.max(spectrograms)

    return spectrograms

# Perform sliding window FFT on each microphone recording.
# Assuming 'recordings' is a 2D array: num_mics x signal_length.

print("Performing sliding window FFT for all microphones")
spectrograms = sliding_window_fft(recordings, fft_window_size, step_size, fs, window_func=None, live_plot=False, num_bins_freq=num_bins_freq, num_bins_time=num_bins_time)

# Preprocess the spectrograms
processed_spectrograms = process_spectrograms(spectrograms, threshold=0)

# Plotting the binned spectrograms for all microphones
num_mics = mic_array.M
signal_length = recordings.shape[1]
plot_binned_spectrograms(processed_spectrograms, fs, signal_length, num_mics)

# # Plotting statistics about the spectrograms
# plot_spectrogram_statistics(processed_spectrograms, fs)

def sliding_window_on_spectrogram(spectrogram, other_spectrograms, search_amp_ms, fs,
                                  window_size_kHz_ms=(0.5, 5), original_duration_s=None, live_plot=False, mse_threshold=1000):
    """
    Process the spectrogram data with a sliding window and optionally plot the spectrogram and focused region.

    Parameters:
        spectrogram (np.array): The spectrogram data.
        other_spectrograms (list of np.array): Other spectrograms to compare.
        search_amp_ms (float): Search amplitude in milliseconds.
        fs (int): Sampling frequency.
        window_size_kHz_ms (tuple): The size of the sliding window in (kHz, ms).
        original_duration_s (float): Original duration of the signal in seconds.
        live_plot (bool): Whether to dynamically plot the spectrogram and focused region.
        mse_threshold(float): Threhold for yielding valid estimations

    Yields:
        float: The central frequency of the window.
        float: The central time of the window.
        list: Time distances to the most similar patches in the other spectrograms.
    """
    num_bins_freq, num_bins_time = spectrogram.shape

    window_freq_kHz, window_time_ms = window_size_kHz_ms
    bin_freq_size_kHz = (fs / 2 / 1000) / num_bins_freq  # size of a frequency bin in kHz
    bin_duration_ms = (original_duration_s * 1000) / num_bins_time  # duration of a bin in ms
    bin_duration_s = bin_duration_ms / 1000

    window_rows = int(window_freq_kHz / bin_freq_size_kHz)
    window_cols = int(window_time_ms / bin_duration_ms)

    # Convert search amplitude from ms to number of bins
    search_amp_bins = int(search_amp_ms / bin_duration_ms)  # converting ms to s and then to bins

    if live_plot:
        plt.ion()
        num_mics = len(other_spectrograms) + 1
        fig, axs = plt.subplots(num_mics, 1, figsize=(12, 8*num_mics))

        # First subplot: original spectrogram
        im = axs[0].imshow(spectrogram, aspect='auto', cmap='inferno', origin='lower',
                           #extent=[0, original_duration_s*1000, 0, fs/2000], interpolation='none')
                           interpolation='none')
        axs[0].set_title("Spectrogram with Sliding Window Focus - Mic 1")
        axs[0].set_xlabel("Time (binned)")
        axs[0].set_ylabel("Frequency (binned)")
        fig.colorbar(im, ax=axs[0], label='Amplitude')
        rect = patches.Rectangle((0, 0), window_cols, window_rows, linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

        # Other subplots: other spectrograms
        other_rects = []
        for idx, other_spectrogram in enumerate(other_spectrograms):
            im_other = axs[idx+1].imshow(other_spectrogram, aspect='auto', cmap='inferno', origin='lower',
                                         #extent=[0, original_duration_s*1000, 0, fs/2000], interpolation='none')
                                         interpolation='none')
            axs[idx+1].set_title(f"Spectrogram with Sliding Window Focus - Mic {idx+2}")
            axs[idx+1].set_xlabel("Time (binned)")
            axs[idx+1].set_ylabel("Frequency (binned)")
            fig.colorbar(im_other, ax=axs[idx+1], label='Amplitude')
            other_rect = patches.Rectangle((0, 0), window_cols, window_rows, linewidth=1, edgecolor='b', facecolor='none')
            axs[idx+1].add_patch(other_rect)
            other_rects.append(other_rect)

    bin_freq_size = fs / 2 / num_bins_freq  # size of a frequency bin in Hz

    for i in range(0, num_bins_time - window_cols + 1, window_cols):
        for j in range(0, num_bins_freq - window_rows + 1, window_rows):
            window = spectrogram[j:j+window_rows, i:i+window_cols]

            if np.amax(window) < 0.1:
                continue

            if live_plot:
                rect.set_xy((i, j))
                plt.pause(0.01)

            # Central frequency of the window
            bottom_freq_bound = j * bin_freq_size
            top_freq_bound = (j + window_rows) * bin_freq_size
            central_freq = (top_freq_bound + bottom_freq_bound) / 2

            # Central time of the window
            left_time_bound = i * bin_duration_s
            right_time_bound = (i + window_cols) * bin_duration_s
            central_time = (left_time_bound + right_time_bound) / 2

            # Find the most similar patch in other spectrograms and calculate time distances
            time_distances = []
            min_mse_values = []  # Keep track of the min_mse values

            for other_idx, other_spectrogram in enumerate(other_spectrograms):
                best_match_j, best_match_i = j, i
                min_mse = np.inf

                for search_i in range(max(0, i-search_amp_bins), min(num_bins_time - window_cols, i+search_amp_bins)):
                    mse = np.mean((window - other_spectrogram[j:j+window_rows, search_i:search_i+window_cols]) ** 2)
                    if mse < min_mse:
                        min_mse, best_match_i = mse, search_i

                min_mse_values.append(min_mse)  # Store the min_mse value

                time_distance = (i - best_match_i) * bin_duration_s
                time_distances.append(time_distance)

                # print()
                # print('search_amp_bins:', search_amp_bins)
                # print('num_bins_time:', num_bins_time)
                # print('original_duration_s:', original_duration_s)
                # print('i:', i)
                # print('best_match_i:', best_match_i)
                # print('bin_duration_s:', bin_duration_s)
                # print('time_distance:', time_distance)

                if live_plot and min_mse < mse_threshold:
                    other_rects[other_idx].set_xy((best_match_i, best_match_j))
                    plt.pause(0.01)

            # Only yield results if all min_mse are below the threshold
            if all(mse_value < mse_threshold for mse_value in min_mse_values):
                yield central_freq, central_time, time_distances


def estimate_source_position_2d(mic_positions, time_diffs, speed_of_sound=343):
    """
    Estimate the position of a sound source given the positions of three microphones and
    time delays between them, assuming a 2D space (XY plane).

    Parameters:
        mic_positions (list of np.array): 3D positions of the microphones.
        time_diffs (list of float): Time delays between mic 1-2 and mic 1-3.
        speed_of_sound (float): Speed of sound in the medium, default is 343 m/s in air.

    Returns:
        np.array: Estimated 2D position of the sound source.
    """

    # Extract microphone positions
    x1, y1, _ = mic_positions[0]
    x2, y2, _ = mic_positions[1]
    x3, y3, _ = mic_positions[2]

    # Extract time differences
    delta_t_21, delta_t_31 = time_diffs

    # Defining the equations to be solved numerically

    def equations(vars):
        x, y, d1 = vars
        eq1 = (x - x2)**2 + (y - y2)**2 - (speed_of_sound * delta_t_21 + d1)**2
        eq2 = (x - x3)**2 + (y - y3)**2 - (speed_of_sound * delta_t_31 + d1)**2
        eq3 = d1**2 - (x - x1)**2 - (y - y1)**2
        return [eq1, eq2, eq3]


    # Initial guess for the source position is the position of the first microphone in the XY plane
    initial_guess = [x1, y1, 0]

    # Solve the system of equations for [x, y]
    solution_numerical = fsolve(equations, initial_guess)

    x,y = solution_numerical[:2]

    # Return estimated position
    return np.array([x,y,0])


# SHOULD RETURN [0, 10, 0]
print(estimate_source_position_2d([[0, 0, 0], [5, 0, 0], [0, 5, 0]], [0.0035, -0.015]))
input()

# List to store all estimated positions
estimated_positions = []

# Extract spectrogram of the first microphone
spectrogram_mic1 = processed_spectrograms[0]

# Extract spectrograms of the other microphones
spectrograms_other_mics = processed_spectrograms[1:]

# Original signal duration in seconds
original_duration_s = signal_length / fs

# Perform sliding window analysis on the first microphone's spectrogram and find similar patches in the other spectrograms
for central_freq, central_time, time_distances in sliding_window_on_spectrogram(
        spectrogram_mic1, spectrograms_other_mics,
        search_amp_ms, fs,
        window_size_kHz_ms, original_duration_s,
        live_plot=False):

    source_position = estimate_source_position_2d(mic_positions, time_distances)

    # print(f"Central Frequency: {central_freq} Hz")
    # print(f"Central Time: {central_time} s")
    time_distances_ms = np.array(time_distances) * 1000
    # print(f"Time Distances: {time_distances_ms} ms")
    # print(f"Estimated position: {source_position}")
    # print()
    # input()

    # Store estimated positions
    estimated_positions.append(source_position)

# Convert to numpy array for easier slicing
estimated_positions = np.array(estimated_positions)

# Plot histograms for x, y, and z coordinates
axis_labels = ['X', 'Y', 'Z']
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, label in enumerate(axis_labels):
    axs[i].hist(estimated_positions[:, i], bins=20, color='skyblue', edgecolor='black')
    axs[i].set_title(f'Histogram of {label} coordinates')
    axs[i].set_xlabel(f'{label} Position')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()

# Extract XY coordinates
xy_coordinates = estimated_positions[:, :2]

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.histplot(x=xy_coordinates[:, 0], y=xy_coordinates[:, 1], bins=20, cmap="YlGnBu", cbar=True, kde=True)

# Formatting
plt.title("Heatmap of Estimated Source Positions (XY-plane, Z=0)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)

plt.show()


