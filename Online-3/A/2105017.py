import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

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

# Load and preprocess the image
image = plt.imread('noisy_image.png')  # Replace with your image file path
# show the image
plt.figure()
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.show()

if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale

image = image / 255.0  # Normalize to range [0, 1]
print (image.shape)

finally_filtered_ft_data = []

sample_rate = 1000

for im in image:
    interval_step = 1  # Adjust for sampling every 'interval_step' data points
    data_sampled = im[::interval_step]
    max_time = len(data_sampled) / (sample_rate / interval_step)
    sampled_times = np.linspace(0, max_time, num=len(data_sampled))

    max_freq = sample_rate / (2 * interval_step)  # max frequency 5512.5
    num_freqs = len(data_sampled)  # number of frequencies is 10334
    frequencies = np.linspace(0, max_freq, num=num_freqs)

    ft_data = fourier_transform(data_sampled, frequencies, sampled_times)

    filtered_ft_data = np.zeros((2, num_freqs))
    filtered_ft_data[0] = ft_data[0].copy()
    filtered_ft_data[1] = ft_data[1].copy()

    # Set the cutoff frequency for the high-pass filter
    cutoff_frequency = 80  # in Hz

    # Apply high-pass filter by zeroing out frequencies below the cutoff
    filtered_ft_data[0][frequencies < cutoff_frequency] = 0
    filtered_ft_data[1][frequencies < cutoff_frequency] = 0

    finally_filtered_ft_data.append(filtered_ft_data)

    # plt.figure(figsize=(12, 6))
    # plt.plot(frequencies, np.sqrt(ft_data[0] ** 2 + ft_data[1] ** 2))
    # plt.title(
    #     "Frequency Spectrum"
    # )
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.show()


denoised_image = []
for f_d in finally_filtered_ft_data:
    filtered_data = inverse_fourier_transform(f_d, frequencies, sampled_times)
    denoised_image.append(filtered_data)

# plt.figure(figsize=(12, 4))
# plt.plot(sampled_times, denoised_image[0])
# plt.title("Reconstructed (Denoised) Audio Signal (Time Domain)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()
# denoised_image = denoised_image * 255

plt.imsave('denoised_image.png', denoised_image, cmap='gray')


plt.figure()
plt.title('Denoised Image')
# plt.imshow(denoised_image, cmap='gray')
# plt.show()
