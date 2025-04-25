import numpy as np


muabo = np.genfromtxt("muabo.txt", delimiter=",") 
muabd = np.genfromtxt("muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 520 # Replace with wavelength in nanometres
blue_wavelength = 465 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf_tissue = 0.01 # Blood volume fraction, average blood amount in tissue
bvf_vein = 1.0 # Blood volume fraction, average blood amount in vein

SO2 = 0.95 # Blood oxygenation

mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*SO2 # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-SO2)) # pure blood

mua_vein = mua_blood*bvf_vein + mua_other
mua_tissue = mua_blood*bvf_tissue + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth

d = 3e-4

pen_depth_vein = np.sqrt(1/(3*(musr+mua_vein)*mua_vein))

C_vein = np.sqrt(3*(musr+mua_vein)*mua_vein)

transmittans_vein = np.exp(-C_vein*d)

reflektans_vein = np.sqrt(3*((musr/mua_vein)+1))

mu_0_vein = 1/(2*pen_depth_vein*mua_vein)


pen_depth_tissue = np.sqrt(1/(3*(musr+mua_tissue)*mua_tissue))

C_tissue = np.sqrt(3*(musr+mua_tissue)*mua_tissue)

transmittans_tissue = np.exp(-C_tissue*d)

reflektans_tissue = np.sqrt(3*((musr/mua_tissue)+1))

mu_0_tissue = 1/(2*pen_depth_tissue*mua_tissue)

kontrast = abs(transmittans_vein - transmittans_tissue)/transmittans_tissue

print(f"Resultater for venen med {bvf_vein*100}% blodfraksjon og {d*1e3}mm dybde:")

print(f"Penetrasjonsdybde: \n Rødt: {pen_depth_vein[0]}, Grønt: {pen_depth_vein[1]}, Blått: {pen_depth_vein[2]}")

print(f"Transmittans: \n Rødt: {transmittans_vein[0]}, Grønt: {transmittans_vein[1]}, Blått: {transmittans_vein[2]}")

print(f"---------------------------------------------------------------------------------------------")

print(f"Resultater for vev med {bvf_tissue*100}% blodfraksjon og {d*1e3}mm dybde:")

print(f"Penetrasjonsdybde: \n Rødt: {pen_depth_tissue[0]}, Grønt: {pen_depth_tissue[1]}, Blått: {pen_depth_tissue[2]}")

print(f"Transmittans: \n Rødt: {transmittans_tissue[0]}, Grønt: {transmittans_tissue[1]}, Blått: {transmittans_tissue[2]}")

print(f"---------------------------------------------------------------------------------------------")

print(f"Kontrast: \n Rødt: {kontrast[0]}, Grønt: {kontrast[1]}, Blått: {kontrast[2]}")

#print(f"Reflektans: \n Rødt: {reflektans_vein[0]}, Grønt: {reflektans_vein[1]}, Blått: {reflektans_vein[2]}")



#z = -np.log(0.5)/C

#print(f"Probet dybde reflektans: \n Rødt: {z[0]}, Grønt: {z[1]}, Blått: {z[2]}")


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

# Resultater med 80% blodfraksjon, 0.1 andel og 1.7mm:
# Penetrasjonsdybde: 
# Rødt: 0.0007255420277254703, Grønt: 0.0002415416153948983, Blått: 0.00017142128829516272
#Transmittans: 
# Rødt: 5.811563140745224e-11, Grønt: 1.7947826683974481e-31, Blått: 4.756471150811384e-44
#Reflektans: 
# Rødt: 6.254116738362698, Grønt: 3.3380483353191734, Blått: 3.1451336470216122
#Probet dybde reflektans: 
# Rødt: 0.0005029074108956554, Grønt: 0.0001674238896988684, Blått: 0.0001188201826697456

# Resultater med 100% blodfraksjon, 0.105 andel og 1.7mm:
#Penetrasjonsdybde: 
# Rødt: 0.0007599340123332098, Grønt: 0.00023515957648749785, Blått: 0.0001634661462190335
#Transmittans:
# Rødt: 1.6885956008826982e-10, Grønt: 2.62779118833407e-32, Blått: 3.706538237798736e-46
#Reflektans:
# Rødt: 6.509047986248299, Grønt: 3.2874335841332982, Blått: 3.0675633917606735
#Probet dybde reflektans:
# Rødt: 0.0005267461180603711, Grønt: 0.00016300019742397995, Blått: 0.00011330609836872282