import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

speed_of_sound = 343  # m/s

# Define room dimensions

# Sampling frequency
#fs=44100
fs=8000

# Create anechoic room (free-field simulation)
room = pra.AnechoicRoom(fs=fs)

def load_sound_source(filepath):
    """
    Load a sound source from a WAV file.

    Parameters:
        filepath (str): Path to the WAV file.

    Returns:
        np.array: Audio signal extracted from the WAV file.
    """
    fs, signal = wavfile.read(filepath)

    # Ensure the signal is mono
    if len(signal.shape) == 2:
        signal = signal[:, 0]

    return fs, signal

def calculate_travel_times(source_positions, mic_positions, speed_of_sound):
    """
    Calculate the time it takes for the signal from each source to reach each microphone.

    ... (rest of the docstring) ...
    """
    num_sources = source_positions.shape[0]
    num_mics = mic_positions.shape[0]
    times = np.zeros((num_sources, num_mics))

    for i, src_pos in enumerate(source_positions):
        for j, mic_pos in enumerate(mic_positions):
            distance = np.sqrt(np.sum((src_pos - mic_pos)**2))
            times[i, j] = distance / speed_of_sound

    return times

# Add sound sources
#source_signals = [np.sin(2 * np.pi * 500 * np.arange(fs * 1) / fs),
#                  np.sin(2 * np.pi * 2000 * np.arange(fs * 1) / fs)]

# Reading the WAV file
fs_source, source_signal_1 = load_sound_source('frequency_sweep_triple_up.wav')
_, source_signal_2 = load_sound_source('frequency_sweep_triple_down.wav')


source_signals = [source_signal_1, source_signal_2]

source_positions = [[0, 0, 0], [10, 0, 0]]

for sig, pos in zip(source_signals, source_positions):
    room.add_source(pos, signal=sig)

# Add microphones
mic_positions = [[1, 1, 0], [1, 2, 0]]
mic_array = pra.MicrophoneArray(np.array(mic_positions).T, room.fs)    # two microphones
room.add_microphone_array(mic_array)

# Ensure numpy arrays are float for division to be accurate
source_positions = np.array(source_positions).astype(float)
mic_positions = np.array(mic_positions).astype(float)

num_sources = source_positions.shape[0]
num_mics = mic_positions.shape[0]

# Initialize a matrix to store times (num_sources x num_mics)
times = calculate_travel_times(source_positions, mic_positions, speed_of_sound)

# Displaying times and time differences in a formatted way
for j, mic_pos in enumerate(mic_positions):
    prev_time = None
    for i, src_pos in enumerate(source_positions):
        time = times[i, j]  # Using the times calculated before
        print(f"Time from source {i+1} to microphone {j+1}: {time*1000:.2f} ms ; {time*fs:.0f} samples")

        # Compute time differences between successive sources for a single microphone
        if prev_time is not None:
            time_diff = time - prev_time
            print(f"Time difference between source {i} and source {i+1} for microphone {j+1}: "
                  f"{time_diff*1000:.2f} ms ; {time_diff*fs:.0f} samples")

        prev_time = time
    print("\n")  # Add a newline for clarity between different microphone outputs

# Compute the RIR (Room Impulse Response)
room.compute_rir()

# Simulate sound propagation
room.simulate()

# Extract the simulated recordings (includes the sound signal, room effects, and any additional noise)
recordings = mic_array.signals

print(recordings.shape)


# Save the recordings to a stereo WAV file
recordings_stereo = np.array(recordings).T  # Transpose to shape (num_samples, num_channels)
wavfile.write('recorded_stereo.wav', fs_source, recordings_stereo)

# Plot the spectrogram for each microphone
window_size = 1024
shift = 5

# Define a figure and axis array for subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns, and setting a figure size

# Iterate through recordings and plot spectrogram
for idx, recording in enumerate(recordings):
    f, t, Sxx = spectrogram(recording, fs=fs_source, nperseg=window_size, noverlap=window_size-shift, scaling='spectrum')
    im = axs[idx].pcolormesh(t, f, 10 * np.log10(Sxx+0.0000000001), shading='gouraud')

    axs[idx].set_ylabel('Frequency [Hz]')
    axs[idx].set_xlabel('Time [sec]')
    axs[idx].set_title(f'Spectrogram for Microphone {idx + 1}')

# Adding a color bar to the right
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Intensity [dB]')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4, right=0.9)
plt.savefig('spectrogram.png', dpi=300)
plt.show()
