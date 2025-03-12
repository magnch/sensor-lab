from utilitites import *

filename = "transmittans_120_måling"

r, g, b = import_rgb(filename)

r_filtered = bandpass_filter(r, 0.5, 4)
g_filtered = bandpass_filter(g, 0.5, 4)
b_filtered = bandpass_filter(b, 0.5, 4)

plot_rgb(r, g, b)
plot_rgb(r_filtered, g_filtered, b_filtered)

freqs, red_fft, green_fft, blue_fft = rgb_fft(r, g, b, N=4096*4)
freqs, red_filtered_fft, green_filtered_fft, blue_filtered_fft = rgb_fft(r_filtered, g_filtered, b_filtered, N=4096*4)
plot_rgb_fft(freqs, red_fft, green_fft, blue_fft)
plot_rgb_fft(freqs, red_filtered_fft, green_filtered_fft, blue_filtered_fft)