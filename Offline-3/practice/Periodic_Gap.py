import numpy as np
import matplotlib.pyplot as plt


# Fourier Transform
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)

    for i in range(num_freqs):
        ft_result_real[i] = np.trapezoid(
            signal * np.cos(2 * np.pi * frequencies[i] * sampled_times), sampled_times
        )
        ft_result_imag[i] = np.trapezoid(
            signal * np.sin(2 * np.pi * frequencies[i] * sampled_times), sampled_times
        )

    return ft_result_real, ft_result_imag


# Inverse Fourier Transform
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    for i in range(n):
        reconstructed_signal[i] = np.trapezoid(
            ft_signal[0] * np.cos(2 * np.pi * frequencies * sampled_times[i])
            + ft_signal[1] * np.sin(2 * np.pi * frequencies * sampled_times[i]),
            frequencies,
        )
    return reconstructed_signal


# Step 1: Generate a periodic signal
sample_rate = 1000  # Hz
duration = 5  # seconds
time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate a sine wave
frequency = 5  # Hz
signal = np.sin(2 * np.pi * frequency * time)

# Introduce gaps (silence) at specific intervals
gap_start = [1, 3]  # Gap start times in seconds
gap_duration = 0.5  # Duration of each gap in seconds
for start in gap_start:
    gap_indices = (time >= start) & (time < start + gap_duration)
    signal[gap_indices] = 0

# Step 2: Visualize the signal with gaps
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label="Signal with Gaps")
plt.title("Periodic Signal with Gaps")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Step 3: Apply Fourier Transform
interval_step = 1
data_sampled = signal[::interval_step]
max_time = len(data_sampled) / (sample_rate / interval_step)
sampled_times = np.linspace(0, max_time, num=len(data_sampled))
max_freq = sample_rate / (2 * interval_step)
num_freqs = len(data_sampled)
frequencies = np.linspace(0, max_freq, num=num_freqs)

# Perform Fourier Transform
ft_real, ft_imag = fourier_transform(data_sampled, frequencies, sampled_times)
magnitude = np.sqrt(ft_real**2 + ft_imag**2)

# Step 4: Visualize the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies, magnitude)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Step 5: Detect gaps by analyzing amplitude dips
reconstructed_signal = inverse_fourier_transform(
    (ft_real, ft_imag), frequencies, sampled_times
)

# Highlight gaps in the reconstructed signal
plt.figure(figsize=(12, 6))
plt.plot(sampled_times, reconstructed_signal, label="Reconstructed Signal")
plt.plot(sampled_times, signal, linestyle="--", label="Original Signal with Gaps")
plt.title("Reconstructed Signal with Gaps Highlighted")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Step 6: Detect gaps using amplitude analysis
threshold = 0.0  # Amplitude threshold to detect gaps
gaps_detected = np.where(np.abs(signal) == threshold)[0] / sample_rate

# Print detected gaps
print(f"Detected Gaps (Seconds): {np.unique(np.floor(gaps_detected))}")
