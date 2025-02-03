import numpy as np
import matplotlib.pyplot as plt
import sys


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.

    Example (requires a recording named `foo.bin`):
    ```
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    ```
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data


# Import data from bin file
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1
            else r'\Users\bryni\Downloads\foo.bin')

sample_time = np.arange(0, sample_period*1000, sample_period)

print(sample_period)
print(data)

data[0] = data[1]

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = data[i][j]*3.3/4096
        data[i][j] += j

data0 = np.zeros(1000)
for i in range(len(data[0])):
    data0[i] = data[0][j]

frequencies = np.arange(0, 1000)
FFT1 = np.fft.fft(data0)


fig, ax = plt.subplots(2)
ax[0].plot(sample_time, data)
ax[1].plot(frequencies, FFT1)
plt.show()


#plt.show()

