from utilitites import *

filename = "radar_test_5.csv"

I, Q = import_radar(filename)
freqs, Sf = radar_fft(I, Q)

f_peak = extract_peak_radar(freqs, Sf)
v = calculate_velocity(f_peak)

print(f"Peak frequency: {f_peak:.2f} Hz")
print(f"Velocity: {v:.2f} m/s")