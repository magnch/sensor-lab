from utilitites import *

filename = "test_puls.csv"
red, green, blue = import_rgb(filename)
plot_rgb(red, green, blue)
freqs, red_fft, green_fft, blue_fft = rgb_fft(red, green, blue)
plot_rgb_fft(freqs, red_fft, green_fft, blue_fft)