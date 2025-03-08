import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
bin_dir = os.path.join(script_dir, "bin")  # Bin folder path
csv_dir = os.path.join(script_dir, "csv")  # CSV folder path
plot_dir = os.path.join(script_dir, "plot")  # Plot folder path
# Ensure directories exists
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)



# Functions 

# Import data produced using adc_sampler.c.
def raspi_import(path, channels=5):

    # Import data produced using adc_sampler.c.
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        data = data.reshape((-1, channels))
        data[0] = data[1]

    # Convert sample period to seconds
    sample_period *= 1e-6
    return sample_period, data

# Import ADC data from csv file, returns time and voltages
def adc_import(filename):
    
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095
    return time, voltages

# Plot imported adc data
def adc_plot(filename, start_ms=0, stop_ms=0, fs=31250):

    time, voltages = adc_import(filename)

    start_sample = int(start_ms * fs / 1e3) if start_ms else None
    stop_sample = int(stop_ms * fs / 1e3) if stop_ms else None

    if start_sample or stop_sample:
        print("Limited time range")
        time = time[start_sample:stop_sample]
        voltages = voltages[start_sample:stop_sample]


    # Plot data
    fig, ax = plt.subplots(5, figsize=(10, 15))
    for i in range(5):
        ax[i].plot(time, voltages[:, i])
        ax[i].set_title(f"Kanal {i + 1}")
        ax[i].set_xlabel("Tid [ms]")
        ax[i].set_ylabel("Amplitude [V]")
        ax[i].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}.png"))

# Perform FFT on imported adc data, all channels
def fft_plot(filename, start_freq=0, stop_freq=-1, padding=0, fs=31250):
    # Import data
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095

    # Print dimension of voltages
    print(voltages.shape)
    

    # Perform FFT
    N = len(voltages[0]) + padding
    FFT = np.fft.fft(voltages, n=N, axis=0)
    freq = np.fft.fftfreq(N, 1/fs)

    # Normalize
    for i in range(5):
        FFT[:, i] = FFT[:, i] / np.max(np.abs(FFT[:, i]))


    # Limit frequency range
    start_index = np.argmax(freq >= start_freq)
    stop_index = np.argmax(freq >= stop_freq)

    # Plot data
    fig, ax = plt.subplots(5, figsize=(10, 15))
    
    for i in range(5):
        ax[i].plot(freq[start_index:stop_index], 10*np.log10(np.abs(FFT[start_index:stop_index, i])))
        ax[i].set_title(f"Kanal {i + 1}")
        ax[i].set_xlabel("Frekvens [Hz]")
        ax[i].set_ylabel("Amplitude [dB]")
        ax[i].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}_FFT.png"))
    print(f"FFT saved to {plot_dir}")  

# Calculate Power Density Spectrum of imported adc data with different windows
def spectrum_window(filename, start_freq=0, stop_freq=-1, padding=0):
    # Import data
    fs = 31250
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095
    ch1 = voltages[:, 0]
    ch1_hamming = np.hamming(len(ch1)) * ch1
    ch1_hann = np.hanning(len(ch1)) * ch1

    # Perform FFT
    N = len(ch1) + padding
    FFT = np.fft.fft(ch1, n=N)
    FFT_hamming = np.fft.fft(ch1_hamming, n=N)
    FFT_hann = np.fft.fft(ch1_hann, n=N)
    freq = np.fft.fftfreq(N, 1/fs)

    # Normalize
    FFT = FFT / np.max(np.abs(FFT))
    FFT_hamming = FFT_hamming / np.max(np.abs(FFT_hamming))
    FFT_hann = FFT_hann / np.max(np.abs(FFT_hann))

    # Limit frequency range
    start_index = np.argmax(freq >= start_freq)
    stop_index = np.argmax(freq >= stop_freq)

    # Plot data
    plt.plot(freq[start_index:stop_index], 20*np.log10(np.abs(FFT[start_index:stop_index])), label="Rektangulær")
    plt.plot(freq[start_index:stop_index], 20*np.log10(np.abs(FFT_hamming[start_index:stop_index])), label="Hamming")
    plt.plot(freq[start_index:stop_index], 20*np.log10(np.abs(FFT_hann[start_index:stop_index])), label="Hann")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title("FFT av kanal 1, med forskjellige vinduer")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}_FFT_window.png"))

# Calculate Power Density Spectrum of imported adc data, only channel 1
def spectrum_ch1(filename, start_freq=0, stop_freq=-1, padding=0):
    # Import data
    fs = 31250
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095
    ch1 = voltages[:, 0]

    # Perform FFT
    N = len(ch1) + padding
    FFT = np.fft.fft(ch1, n=N)
    freq = np.fft.fftfreq(N, 1/fs)

    # Normalize
    FFT = FFT / np.max(np.abs(FFT))

    # Limit frequency range to positive values
    freq = freq[:N//2]
    FFT = FFT[:N//2]

    # Plot data
    plt.figure(figsize=(10, 5))
    plt.plot(freq, 20*np.log10(np.abs(FFT)))
    plt.ylabel("Amplitude [dB]")
    plt.title("Effektspekter av målt signal på kanal 1")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}_spekter.png"))

# Plot magnitude reponse of filter
def magnitude_plot(filename, start_freq=0, stop_freq=16e3):
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    f = data[:, 0]
    V_in = data[:, 1]
    V_out = data[:, 2]

    # Limit frequency range
    start_index = np.argmax(f >= start_freq)
    stop_index = np.argmax(f >= stop_freq)

    # Plot data
    plt.figure()
    plt.semilogx(f[start_index:stop_index], 20*np.log10(V_out[start_index:stop_index]/V_in[start_index:stop_index]))
    plt.title("Bode plot")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}_bode.png"))

# Apply window function to imported adc data
def window_csv(filename, window=0):
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095

    for i in range(5):
        if window:
            voltages[:,i] = np.hamming(len(voltages)) * voltages[:,i]
            win = "Hamming"
        else:
            voltages[:,i] = np.hanning(len(voltages)) * voltages[:,i]
            win = "Hann"
    
    #Export to csv
    np.savetxt(
        csv_path[:-4] + win + ".csv",
        np.column_stack((time, voltages)),
        delimiter=',',
        header='Time, Channel 1, Channel 2, Channel 3, Channel 4, Channel 5',
        comments='',
        fmt='%1.4e'
    )

    print(f"Data saved to {csv_path}")

# Returns the cross-correlation of two signals, and an array of lags
def cross_correlate(x, y, start_sample=200, limit=0):
    if start_sample:
        x = x[start_sample:]
        y = y[start_sample:]


    r_xy = np.correlate(x, y, mode="full")
    r_xy /= np.max(np.abs(r_xy)) # Normalize
    lags = np.arange(-len(x) + 1, len(y))

    if limit:
        center = len(lags) // 2
        lags = lags[center - limit:center + limit + 1]
        r_xy = r_xy[center - limit:center + limit + 1]
        
    return r_xy, lags

# Plot the cross-correlation of two signals, with max value and lag
def plot_correlation(x, y, start_sample=200, limit=0):
    r_xy, lags = cross_correlate(x, y, start_sample)

    # Limit number of samples around center
    if limit:
        center = len(lags) // 2
        lags = lags[center - limit:center + limit + 1]
        r_xy = r_xy[center - limit:center + limit + 1]

    max_value = np.max(np.abs(r_xy))
    max_sample = lags[np.argmax(np.abs(r_xy))]


    plt.figure()
    plt.plot(lags, r_xy)
    plt.plot(max_sample, max_value, "ro", label=f"Max value: {max_value:.2f} at l = {max_sample}")
    plt.title("Cross-correlation")
    plt.xlabel("Lag [samples]")
    plt.ylabel("r_xy")
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot the cross-correlation of all three microphones, with max values and lags
def plot_correlation_all(m1, m2, m3, limit=0, start_sample=200):


    # Calculate cross-correlation
    r12, lags = cross_correlate(m1, m2, start_sample)
    r13, lags = cross_correlate(m1, m3, start_sample)
    r23, lags = cross_correlate(m2, m3, start_sample)

    # Limit number of samples around center
    if limit:
        center = len(lags) // 2
        lags = lags[center - limit:center + limit + 1]
        r12 = r12[center - limit:center + limit + 1]
        r13 = r13[center - limit:center + limit + 1]
        r23 = r23[center - limit:center + limit + 1]

    # Find max values and lags
    max12 = np.max(np.abs(r12))
    max13 = np.max(np.abs(r13))
    max23 = np.max(np.abs(r23))

    max_sample12 = lags[np.argmax(np.abs(r12))]
    max_sample13 = lags[np.argmax(np.abs(r13))]
    max_sample23 = lags[np.argmax(np.abs(r23))]

    # Plot
    fig, ax = plt.subplots(3, figsize=(10, 15))
    ax[0].plot(lags, r12)
    ax[0].plot(max_sample12, max12, "ro", label=f"Max value: {max12:.2f} at l = {max_sample12}")
    ax[0].set_title("Cross-correlation between M1 and M2")
    ax[0].set_xlabel("Lag [samples]")
    ax[0].set_ylabel("r_12")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(lags, r13)
    ax[1].plot(max_sample13, max13, "ro", label=f"Max value: {max13:.2f} at l = {max_sample13}")
    ax[1].set_title("Cross-correlation between M1 and M3")
    ax[1].set_xlabel("Lag [samples]")
    ax[1].set_ylabel("r_13")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(lags, r23)
    ax[2].plot(max_sample23, max23, "ro", label=f"Max value: {max23:.2f} at l = {max_sample23}")
    ax[2].set_title("Cross-correlation between M2 and M3")
    ax[2].set_xlabel("Lag [samples]")
    ax[2].set_ylabel("r_23")
    ax[2].grid()
    ax[2].legend()

    plt.tight_layout()
    path = os.path.join(plot_dir, "cross_correlation.png")
    plt.savefig(path)

# Calculate delay in samples between two signals using cross-correlation
def calculate_delay(x, y, start_sample = 200, limit=0):
    r_xy, samples = cross_correlate(x, y, start_sample=200, limit=limit)
    max_sample = samples[np.argmax(np.abs(r_xy))]
    #print(f"Delay in samples: {max_sample}")
    return max_sample

# Calculate delay in milliseconds between two signals using cross-correlation
def calculate_delay_ms(x, y, fs=31250, start_sample=200):
    return calculate_delay(x, y) * (1000/fs)

# Calculates the incident angle of an incoming sound wave, in degrees
def calculate_angle(m1, m2, m3, start_sample=200):
    n21 = calculate_delay(m2, m1, limit=5)
    n31 = calculate_delay(m3, m1, limit=5)
    n32 = calculate_delay(m3, m2, limit=5)
    den = (n31- n21 + 2*n32)
    num = np.sqrt(3) * (n31 + n21)
    angle = np.arctan2(num, den) * (180/np.pi)

    angle += 180
    if angle >= 360:
        angle -= 360

    return  angle