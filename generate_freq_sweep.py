import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_freq_sweep(start_freq, end_freq, duration, sample_rate):
    """
    Generate a frequency sweep signal.

    Parameters:
    - start_freq: Initial frequency of the sweep (Hz).
    - end_freq: Final frequency of the sweep (Hz).
    - duration: Duration of the sweep (seconds).
    - sample_rate: Sampling rate of the generated signal (Hz).

    Returns:
    - signal: NumPy array containing the generated signal.
    """

    if start_freq >= sample_rate/2 or end_freq >= sample_rate/2:
        raise ValueError("Frequencies must be lower than half the sampling rate (Nyquist limit).")

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t**2 / duration)
    signal = 0.5 * np.sin(phase)

    # Applying a Hann window to smooth the signal at the edges.
    window = np.hanning(len(t))
    signal = signal * window

    return signal



# Parameters
sample_rate = 44100  # in Hz
duration = 0.2  # in seconds

# Define a list of tuples for multiple sweeps. Each tuple contains (start_freq, end_freq)
#sweeps = [(5000, 1000), (10000, 5000), (15000, 10000)]
#sweep_name = 'frequency_sweep_triple_down.wav'
sweeps = [(1000, 5000), (5000, 10000), (10000, 15000)]
sweep_name = 'frequency_sweep_triple_up.wav'

# Generate signals and sum them
signals = np.array([generate_freq_sweep(start, end, duration, sample_rate) for start, end in sweeps])
signal = np.sum(signals, axis=0)


# Normalize the signal
signal = signal / np.max(np.abs(signal))

# Convert to 16-bit PCM format
signal_pcm = np.int16(signal * 32767)

# Save to a wave file
wavfile.write(sweep_name, sample_rate, signal_pcm)

# Visual Inspection (Spectrogram)
f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=1024, noverlap=512, scaling='spectrum')
plt.pcolormesh(t, f, 10 * np.log10(Sxx+1e-15), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.show()
