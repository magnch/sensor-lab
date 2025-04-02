import numpy as np


muabo = np.genfromtxt("handout-code\\muabo.txt", delimiter=",") 
muabd = np.genfromtxt("handout-code\\muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 520 # Replace with wavelength in nanometres
blue_wavelength = 465 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth

pen_depth = np.sqrt(1/(3*(musr+mua)*mua))

d = 0.0171
#d = 3e-4

C = np.sqrt(3*(musr+mua)*mua)

print(f"Penetrasjonsdybde: \n Rødt: {pen_depth[0]}, Grønt: {pen_depth[1]}, Blått: {pen_depth[2]}")

transmittans = np.exp(-C*d)

print(f"Transmittans: \n Rødt: {transmittans[0]}, Grønt: {transmittans[1]}, Blått: {transmittans[2]}")

reflektans = np.sqrt(3*((musr/mua)+1))

print(f"Reflektans: \n Rødt: {reflektans[0]}, Grønt: {reflektans[1]}, Blått: {reflektans[2]}")

mu_0 = 1/(2*pen_depth*mua)

z = -np.log(0.5)/C

print(f"Probet dybde reflektans: \n Rødt: {z[0]}, Grønt: {z[1]}, Blått: {z[2]}")


# Transmittans-resultater med 100% blodfraksjon og 300um:
# 15.2% 0.13% 0.012%
#
# Transmittans-resultater med 1% blodfraksjon og 300um:
# 82.6% 69.8% 63.0%
#
# Kontrast
# 1:5.4 1:536.9 1:5250
#
# Blått kommer til å fungere mest