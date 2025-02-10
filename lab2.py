import numpy as np
import matplotlib.pyplot as plt

'''
Legg inn kode for å importere data fra csv
Bruker midlertidige dummy arrays
'''
fs = 1000

t = np.linspace(0, 1, fs)
x = np.sin(2 * np.pi * 10 * t)
y = np.cos(2 * np.pi * 10 * t)

def calculate_delay(x, y, fs):

    r_xy = np.correlate(x, y, mode="full")
    samples = np.arange(-len(x) + 1, len(y))
    max_sample = samples[np.argmax(np.abs(r_xy))]
    delay = max_sample / fs
    return delay



print(calculate_delay(x, y, fs))