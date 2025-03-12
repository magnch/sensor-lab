from utilitites import *

filebase = "transmittans_måling_"

rgb_peaks = np.zeros((5, 3))
filter = True

# Loop over all files, plot RGB values and FFT for each 
for i in range(5):
    filename = filebase + str(i+1) + ".csv"
    r, g, b = import_rgb(filename)

    # Apply bandpass filter
    if filter:
        r = bandpass_filter(r)
        g = bandpass_filter(g)
        b = bandpass_filter(b)

    freqs, r_f, g_f, b_f = rgb_fft(r, g, b)
    plot_rgb(r, g, b, filename)
    print(f"Data for {filename} plotted.")
    plot_rgb_fft(freqs, r_f, g_f, b_f, filename, f_min=0.5, f_max=2.5)
    print(f"FFT for {filename} plotted.")
    print("---------------------------------------------------")
    # Extract peak values for each color
    rgb_peaks[i] = extract_peak_rgb(freqs, r_f, g_f, b_f, f_min=0.5, f_max=2.5)

# Loop over all robusthet files, plot RGB values and FFT for each
for maling in ["lys", "hoy_puls"]:
    filename = "robusthet_" + maling + ".csv" 
    r, g, b = import_rgb(filename)

    if filter:
        r = bandpass_filter(r)
        g = bandpass_filter(g)
        b = bandpass_filter(b)

    freqs, r_f, g_f, b_f = rgb_fft(r, g, b)
    plot_rgb(r, g, b, filename)
    print(f"Data for {filename} plotted.")
    plot_rgb_fft(freqs, r_f, g_f, b_f, filename)
    print(f"FFT for {filename} plotted.")
    print("---------------------------------------------------")

# Calculate mean and std for each color
for i, color in enumerate(["R", "G", "B"]):
    # Calculate mean and std for each color
    mean, std = calculate_mean_and_std(rgb_peaks[:, i])
    print(f"Mean for {color}: {mean:.2f}")
    print(f"Std for {color}: {std:.2f}")

