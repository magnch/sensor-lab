import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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

    if start_sample and stop_sample:
        print("Limited time range")
        time = time[start_sample:stop_sample]
        voltages = voltages[start_sample:stop_sample]


    # Plot data
    fig, ax = plt.subplots(5, figsize=(10, 15))
    for i in range(5):
        ax[i].plot(time, voltages[:, i])
        ax[i].set_title(f"Channel {i + 1}")
        ax[i].set_xlabel("Time [ms]")
        ax[i].set_ylabel("Amplitude [V]")
        ax[i].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{filename[:-4]}.png"))


# Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
csv_dir = os.path.join(script_dir, "csv")  # CSV folder path
plot_dir = os.path.join(script_dir, "plot")  # Plot folder path
# Ensure CSV directory exists
os.makedirs(plot_dir, exist_ok=True)

filename = sys.argv[1] if len(sys.argv) > 1 else "test.csv"
start_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 0
stop_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 0
adc_plot(filename, start_ms, stop_ms)