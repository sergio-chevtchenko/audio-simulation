import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_freq_sweep(start_freq, end_freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t**2 / duration)
    signal = 0.5 * np.sin(phase)
    return signal


# Parameters
sample_rate = 44100  # in Hz
duration = 1.0  # in seconds

# Define a list of tuples for multiple sweeps. Each tuple contains (start_freq, end_freq)
#sweeps = [(5000, 1000), (10000, 5000), (15000, 10000)]
sweeps = [(1000, 5000), (5000, 10000), (10000, 15000)]

# Generate signals and sum them
signals = np.array([generate_freq_sweep(start, end, duration, sample_rate) for start, end in sweeps])
signal = np.sum(signals, axis=0)


# Normalize the signal
signal = signal / np.max(np.abs(signal))

# Convert to 16-bit PCM format
signal_pcm = np.int16(signal * 32767)

# Save to a wave file
wavfile.write('frequency_sweep_triple_up.wav', sample_rate, signal_pcm)

# Visual Inspection (Spectrogram)
f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=1024, noverlap=512, scaling='spectrum')
plt.pcolormesh(t, f, 10 * np.log10(Sxx+1e-15), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.show()
