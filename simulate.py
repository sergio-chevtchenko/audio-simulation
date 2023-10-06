import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

speed_of_sound = 343  # m/s

# Define room dimensions
room_dim = [10, 10, 3]  # width, length, height

fs=44100

# Create a shoebox room
room = pra.ShoeBox(room_dim, fs=fs, max_order=2)

# Add sound sources
#source_signals = [np.sin(2 * np.pi * 500 * np.arange(fs * 1) / fs),
#                  np.sin(2 * np.pi * 2000 * np.arange(fs * 1) / fs)]

# Reading the WAV file
fs, source_signal_1 = wavfile.read('C1.wav')
_, source_signal_2 = wavfile.read('C1.wav')

# Check if the signals have two channels, and if so, use only the first channel
if len(source_signal_1.shape) == 2:
    source_signal_1 = source_signal_1[:, 0]
if len(source_signal_2.shape) == 2:
    source_signal_2 = source_signal_2[:, 0]

source_signals = [source_signal_1, source_signal_2]

source_positions = [[2, 9, 1], [9, 2, 1]]

for sig, pos in zip(source_signals, source_positions):
    room.add_source(pos, signal=sig)

# Add microphones
mic_positions = [[1, 9, 1], [9, 1, 1]]
mic_array = pra.MicrophoneArray(np.array(mic_positions).T, room.fs)    # two microphones
room.add_microphone_array(mic_array)

# Ensure numpy arrays are float for division to be accurate
source_positions = np.array(source_positions).astype(float)
mic_positions = np.array(mic_positions).astype(float)

num_sources = source_positions.shape[0]
num_mics = mic_positions.shape[0]

# Initialize a matrix to store times (num_sources x num_mics)
times = np.zeros((num_sources, num_mics))

# Calculate the time it takes for the signal from each source to reach each microphone
for i, src_pos in enumerate(source_positions):
    for j, mic_pos in enumerate(mic_positions):
        distance = np.sqrt(np.sum((src_pos - mic_pos)**2))
        times[i, j] = distance / speed_of_sound

# Displaying times in a formatted way
for i, src_times in enumerate(times):
    for j, time in enumerate(src_times):
        print(f"Time from source {i+1} to microphone {j+1}: {time*1000:.2f} ms ; {time*fs:.0f} samples")

# Compute the RIR (Room Impulse Response)
room.compute_rir()

# Simulate sound propagation
room.simulate()

# Extract the simulated recordings (includes the sound signal, room effects, and any additional noise)
recordings = mic_array.signals

print(recordings.shape)


# Save the recordings to a stereo WAV file
recordings_stereo = np.array(recordings).T  # Transpose to shape (num_samples, num_channels)
wavfile.write('recorded_stereo.wav', fs, recordings_stereo)

# Plot the spectrogram for each microphone
window_size = 256
shift = int(window_size*0.5)

# Define a figure and axis array for subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns, and setting a figure size

# Iterate through recordings and plot spectrogram
for idx, recording in enumerate(recordings):
    f, t, Sxx = spectrogram(recording, fs=fs, nperseg=window_size, noverlap=window_size-shift, scaling='spectrum')
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
