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

vinkler = [0, 90, 210]
plot = False

for vinkel in vinkler:

    measurements = np.array([])
    print(f"Measuring for {vinkel} degrees:")
    print("--------------------------------")
    for i in range(1, 6):

        filename = f"måling_{vinkel}_{i}.csv"
        limit = 7
        start_sample = 200

        time, voltage = adc_import(filename)
        # Remove DC offset
        voltage = voltage - 1.65
        samples = np.arange(0, len(voltage))

        m1 = voltage[:, 0]
        m2 = voltage[:, 1]
        m3 = voltage[:, 3]

        if not plot:
            plot_correlation_all(m1, m2, m3, limit=7)
            plot_correlation(m1, m1, limit=7)
            plot = True

        calculated_angle = calculate_angle(m1, m2, m3)

        #plot_correlation_all(m1, m2, m3, limit, start_sample)
        print(f"Measurement {i}: {calculated_angle}")
        measurements = np.append(measurements, calculated_angle)

    sample_mean = np.mean(measurements)
    sample_var = np.var(measurements)
    print(f"Sample mean for {vinkel} degrees: {sample_mean}")
    print(f"Sample variance for {vinkel} degrees: {sample_var}")
    print("---------------------------------------------------")