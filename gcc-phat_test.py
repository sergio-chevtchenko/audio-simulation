import numpy as np
import matplotlib.pyplot as plt



nSample = int(6e3)
signalNoisePower = 0.001
timeNoisePower = 0.00000001
ts = np.linspace(0, 1, nSample)

# Function to generate the signal
def generate_signal(w, ts, timeNoisePower, signalNoisePower, timeSlope):
    noisyTime = np.convolve(ts + timeNoisePower * np.random.randn(len(ts)), np.ones(100)/100, 'same')
    y = np.sin(noisyTime * 2 * np.pi * w * 2/3. * timeSlope)
    noisyTime = np.convolve(ts + timeNoisePower * np.random.randn(len(ts)), np.ones(100)/100, 'same')
    y += np.sin(noisyTime * 2 * np.pi * w * timeSlope)
    noisyTime = np.convolve(ts + timeNoisePower * np.random.randn(len(ts)), np.ones(100)/100, 'same')
    y += np.sin(noisyTime * 2 * np.pi * w * 4/5. * timeSlope)
    return y

timeSlope = 0.1 * ts + 0.9

# Generate signals y0 and y1
w0 = 900.57645645
y0 = generate_signal(w0, ts, timeNoisePower, signalNoisePower, timeSlope)

y0Mask = 0.5 * (1 - np.cos(ts * 3 * np.pi))
y0Mask[int(2/3 * nSample):] = 0
y0 *= y0Mask
y0 /= np.std(y0)
y0 += signalNoisePower * np.random.randn(len(y0))

D0 = 50
y0D = np.roll(y0, D0)

w1 = 650.785432
y1 = generate_signal(w1, ts, timeNoisePower, signalNoisePower, timeSlope)

y1Mask = 0.5 * (1 - np.cos(ts * 3 * np.pi + np.pi))
y1Mask[:int(nSample/3)] = 0
y1 *= y1Mask
y1 /= np.std(y1)
y1 += signalNoisePower * np.random.randn(len(y1))
y1 *= 1

D1 = -5
y1D = np.roll(y1, D1)

# Mixing signals
signal = y0 + y1
signal_delayed = y0D+y1D

# Parameters
fs = 16000  # Sampling rate [Hz]
n_samples = nSample  # Total samples
delay = D0  # Delay introduced in samples


# fs = 16000  # Sampling rate [Hz]
# n_samples = 16000  # Total samples
# delay = 1000  # Delay introduced in samples

# # Create signals
# np.random.seed(1)
# signal = np.random.randn(n_samples)  # Random noise signal
# signal_delayed = np.concatenate((np.zeros(delay), signal[:-delay]))

def gcc_phat(sig1, sig2, fs, max_tau=None, interp=1, visualize=False):
    """
    This function computes the offset between the signal sig1 and the reference signal sig2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.

    Parameters:
    - sig1: array_like, the first signal
    - sig2: array_like, the second signal (reference signal)
    - fs: int, sampling frequency
    - max_tau: float, maximum expected delay (in seconds) between the two signals
    - interp: int, interpolation factor to improve peak detection
    - visualize: bool, whether to generate step-by-step visualizations

    Returns:
    - offset: int, estimated offset between sig1 and sig2
    """

    # Compute Fourier Transform of signals
    SIG1 = np.fft.rfft(sig1, n=interp*sig1.size)
    SIG2 = np.fft.rfft(sig2, n=interp*sig2.size)

    # Compute the PHAT weighting
    W = 1 / (np.abs(SIG1) * np.abs(SIG2))

    # Compute the cross-correlation function
    R = np.fft.irfft(W * SIG1.conjugate() * SIG2, n=interp*sig1.size)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), int(interp * sig1.size/2))
        shifts = np.concatenate((np.arange(-max_shift, 0), np.arange(interp*sig1.size - max_shift, interp*sig1.size)))
        R = R[shifts]
    else:
        shifts = np.arange(-interp*sig1.size/2, interp*sig1.size/2)

    offset = shifts[np.argmax(R)]

    if visualize:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.title('Signals in Time Domain')
        plt.plot(sig1, label='Sig1')
        plt.plot(sig2, label='Sig2')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.title('Spectra of the Signals')
        plt.plot(np.abs(SIG1), label='|FFT(Sig1)|')
        plt.plot(np.abs(SIG2), label='|FFT(Sig2)|')
        plt.xlabel('Frequency Bin Index')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.title('PHAT Weighting')
        plt.plot(W, label='PHAT Weight')
        plt.xlabel('Frequency Bin Index')
        plt.ylabel('Weight')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title('Cross-Correlation Function')
        plt.plot(shifts, R, label='R')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return offset / interp


# Estimate the delay
estimated_delay = gcc_phat(signal_delayed, signal, fs=fs, max_tau=0.1, visualize=True)

# Display the results
print(f"Actual delay: {delay} samples")
print(f"Estimated delay: {int(estimated_delay)} samples")

# Plot the signals
plt.figure(figsize=(10, 6))
time_vector = np.arange(n_samples) / fs
plt.plot(time_vector, signal, label='Original Signal')
plt.plot(time_vector, signal_delayed, label='Delayed Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Original and Delayed Signals')
plt.grid(True)
plt.show()
