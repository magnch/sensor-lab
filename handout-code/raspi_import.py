import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.
    """
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        data = data.reshape((-1, channels))
        data[0] = data[1]

    # Convert sample period to seconds
    sample_period *= 1e-6
    return sample_period, data


# Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))[:-12]  # Get script directory
bin_dir = os.path.join(script_dir, "bin")  # Bin folder path
csv_dir = os.path.join(script_dir, "csv")  # CSV folder path

# Ensure CSV directory exists
os.makedirs(csv_dir, exist_ok=True)

# Get the first argument or use a default filename
filename = sys.argv[1] if len(sys.argv) > 1 else "test.bin"
bin_path = os.path.join(bin_dir, filename)

if __name__ == "__main__":
    sample_period, data = raspi_import(bin_path)

    # Generate time array
    sample_time = np.arange(0, sample_period * len(data), sample_period)

    # Construct CSV file path
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    # Export data to CSV
    np.savetxt(
        csv_path,
        np.column_stack((sample_time, data)),
        delimiter=',',
        header='Time, Channel 1, Channel 2, Channel 3, Channel 4, Channel 5',
        comments='',
        fmt='%1.4e'
    )

    print(f"Data saved to {csv_path}")
