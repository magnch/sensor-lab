from utilitites import *

'''
Legg inn kode for å importere data fra csv
Bruker midlertidige dummy arrays
'''
fs = 1000

t = np.linspace(0, 0.1, fs)
x = np.sin(2 * np.pi * 10 * t)
y = np.cos(2 * np.pi * 10 * t)

# Plot cross-correlation
plot_correlation(x, y, fs)


print(calculate_delay(x, y, fs))