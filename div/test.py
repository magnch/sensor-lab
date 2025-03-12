from utilitites import *

filter = False
filename = "test_walter.csv"

r, g, b = import_rgb(filename)

if filter:
    r = bandpass_filter(r)
    g = bandpass_filter(g)
    b = bandpass_filter(b)

freqs, r_f, g_f, b_f = rgb_fft(r, g, b)
plot_rgb_fft(freqs, r_f, g_f, b_f, save=False)
print(extract_peak_rgb(freqs, r_f, g_f, b_f))