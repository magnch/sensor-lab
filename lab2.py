from utilitites import *

'''
Mikrofoner koblet inn på ADC 1, 2 og 4 - GPIO 23, 24 og 26
Start med lave frekvenser [400, 5000]
Sjekk frekvenser helt ned til øvre og nedre grenser 


Avstand mellom mikrofoner: 5.3 cm
Lydhastighet: 343 m/s
Maks samples: 5 (4,83)

'''

'''
Legg inn kode for å importere data fra csv
Bruker midlertidige dummy arrays
'''
""" fs = 1000

t = np.linspace(0, 0.1, fs)
x = np.sin(2 * np.pi * 10 * t)
y = np.cos(2 * np.pi * 10 * t)

# Plot cross-correlation
plot_correlation(x, y, fs)


print(calculate_delay(x, y, fs)) """

filename = "mic_check_400.csv"
limit = 7
start_sample = 200

time, voltage = adc_import(filename)
# Remove DC offset
voltage = voltage - 1.65
samples = np.arange(0, len(voltage))

m1 = voltage[:, 0]
m2 = voltage[:, 1]
m3 = voltage[:, 3]

#plot_correlation_all(m1, m2, m3, limit, start_sample)
print(calculate_angle(m1, m2, m3))