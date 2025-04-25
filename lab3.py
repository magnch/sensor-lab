from utilitites import *

filebase = "reflektans_rapport_måling_"

rgb_peaks = np.zeros((5, 3))
filter = True
robusthet = True
save = True
ref_vals = np.array([[72, 73, 73, 73.5, 76],
                    [64, 63, 63.5, 65, 66.5]]) / 60

robusthet_ref_vals = np.array([61.5, 123]) / 60

f_low = 0.75
f_high = 4


def plot_all():

    # Loop over all files, plot RGB values and FFT for each 
    for i in range(5):
        filename = filebase + str(i+1) + ".csv"
        r, g, b = import_rgb(filename)

        # Apply bandpass filter
        if filter:
            r = bandpass_filter(r, f_min=f_low, f_max=f_high)
            g = bandpass_filter(g, f_min=f_low, f_max=f_high)
            b = bandpass_filter(b, f_min=f_low, f_max=f_high)

        

        freqs, r_f, g_f, b_f = rgb_fft(r, g, b)
        plot_rgb(r, g, b, filename=filename, save=save)
        print(f"Data for {filename} plotted.")
        plot_rgb_fft(freqs, r_f, g_f, b_f, filename=filename, save=save, f_min=f_low, f_max=f_high)
        print(f"FFT for {filename} plotted.")
        print("---------------------------------------------------")
        # Extract peak values for each color
        rgb_peaks[i] = extract_peak_rgb(freqs, r_f, g_f, b_f, f_min=f_low, f_max=f_high)
        print(f"Peak values: {rgb_peaks[i]}")
        calculate_snr_rgb(r_f, g_f, b_f, freqs, f_signal=ref_vals[1][i])


    if robusthet:
        # Loop over all robusthet files, plot RGB values and FFT for each
        for index, maling in enumerate(["lys", "hoy_puls"]):
            filename = "robusthet_" + maling + "_limited.csv" 
            r, g, b = import_rgb(filename)

            if filter:
                r = bandpass_filter(r, f_min=f_low, f_max=f_high)
                g = bandpass_filter(g, f_min=f_low, f_max=f_high)
                b = bandpass_filter(b, f_min=f_low, f_max=f_high)


            freqs, r_f, g_f, b_f = rgb_fft(r, g, b)
            calculate_snr_rgb(r_f, g_f, b_f, freqs, f_signal=robusthet_ref_vals[index])
            plot_rgb(r, g, b, filename=filename, save=save)
            
            print(f"Data for {filename} plotted.")
            plot_rgb_fft(freqs, r_f, g_f, b_f, filename=filename, save=save, f_min=f_low, f_max=f_high)
            print(f"FFT for {filename} plotted.")
            print(extract_peak_rgb(freqs, r_f, g_f, b_f, f_min=f_low, f_max=f_high))
            print("---------------------------------------------------")

def plot_raw():
    # Loop over all files, plot RGB values and FFT for each 
    for i in range(5):
        filename = filebase + str(i+1) + ".csv"
        r, g, b = import_rgb(filename)
        filename = filebase + str(i+1) + "_raw.png"

        plot_rgb(r, g, b, filename=filename, save=save)
        print(f"Data for {filename} plotted.")
        print("---------------------------------------------------")
    
    if robusthet:
        # Loop over all robusthet files, plot RGB values and FFT for each
        for index, maling in enumerate(["lys", "hoy_puls"]):
            filename = "robusthet_" + maling + "_limited.csv" 
            r, g, b = import_rgb(filename)
            filename = filename[:-4] + "_raw.png"

            plot_rgb(r, g, b, filename=filename, save=save)
            print(f"Data for {filename} plotted.")
            print("---------------------------------------------------")

def mean_std_all():
    # Calculate mean and std for each color
    for i, color in enumerate(["R", "G", "B"]):
    # Calculate mean and std for each color
        mean, std = calculate_mean_and_std(rgb_peaks[:, i])
        print(f"Mean for {color}: {mean:.2f}")
        print(f"Std for {color}: {std:.2f}")


plot_all()