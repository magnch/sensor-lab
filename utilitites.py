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

# Plot imported adc data
def adc_plot(filename, start_ms=0, stop_ms=0):
    # Import data
    fs = 31250
    csv_path = os.path.join(csv_dir, filename)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    time = data[:, 0] * 1e3  # Convert to ms
    channels = data[:, 1:]
    voltages = channels * 3.3 / 4095
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