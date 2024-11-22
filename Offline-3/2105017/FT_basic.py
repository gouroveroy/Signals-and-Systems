import numpy as np
import matplotlib.pyplot as plt


# Define the interval and function and generate appropriate x values and y values
interval = (-2, 2)


def function(x_values, funcType="Parabolic"):
    if funcType == "Parabolic":
        return np.where(
            (x_values >= interval[0]) & (x_values <= interval[1]), x_values**2, 0
        )

    if funcType == "Triangular":
        return np.where(
            (x_values >= interval[0]) & (x_values <= interval[1]),
            1 - np.abs(x_values / 2),
            0,
        )

    if funcType == "Sawtooth":
        T = interval[1] - interval[0]
        return np.where(
            (x_values >= interval[0]) & (x_values <= interval[1]),
            (x_values - interval[0]) % T,
            0,
        )

    if funcType == "Rectangular":
        return np.where((x_values >= interval[0]) & (x_values <= interval[1]), 1, 0)


x_values = np.linspace(-10, 10, 1000)


# Fourier Transform
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)

    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT
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
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal
    for i in range(n):
        reconstructed_signal[i] = np.trapezoid(
            ft_signal[0] * np.cos(2 * np.pi * frequencies * sampled_times[i])
            + ft_signal[1] * np.sin(2 * np.pi * frequencies * sampled_times[i]),
            frequencies,
        )

    return reconstructed_signal


frequency_ranges = [(-1, 1), (-2, 2), (-5, 5)]
function_types = ["Parabolic", "Triangular", "Sawtooth", "Rectangular"]
# function_types = ["Sawtooth"]

for funcType in function_types:
    y_values = function(x_values, funcType)

    # Plot the original function
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, y_values, label=f"Original {funcType} function")
    plt.title(f"Original Function ({funcType})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    for freq_range in frequency_ranges:
        # Define the sampled times and frequencies
        sampled_times = x_values
        frequencies = np.linspace(freq_range[0], freq_range[1], 1000)

        # Apply FT to the sampled data
        ft_data = fourier_transform(y_values, frequencies, sampled_times)

        # Plot the FT data
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, np.sqrt(ft_data[0] ** 2 + ft_data[1] ** 2))
        plt.title(
            f"Frequency Spectrum of {funcType} function (Frequency Range: {freq_range})"
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

    for freq_range in frequency_ranges:
        # Define the sampled times and frequencies
        sampled_times = x_values
        frequencies = np.linspace(freq_range[0], freq_range[1], 1000)

        # Apply FT to the sampled data
        ft_data = fourier_transform(y_values, frequencies, sampled_times)

        # Reconstruct the signal from the FT data
        reconstructed_y_values = inverse_fourier_transform(
            ft_data, frequencies, sampled_times
        )

        # Plot the original and reconstructed functions for comparison
        plt.figure(figsize=(12, 4))
        plt.plot(
            x_values, y_values, label=f"Original {funcType} function", color="blue"
        )
        plt.plot(
            sampled_times,
            reconstructed_y_values,
            label=f"Reconstructed {funcType} function",
            color="red",
            linestyle="--",
        )
        plt.title(
            f"Original vs Reconstructed Function ({funcType}, Frequency Range: {freq_range})"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
