import numpy as np

# https://www.electronicshub.org/active-band-pass-filter/

def R(fc, C):
    return 1/(2*np.pi* fc*C)

def C(fc, A, R):
    return 1/(2*np.pi*fc*R)

fc_low = 2.2
fc_high = 2.3e3
C2 = 1e-9
A = 10
R1 = R(fc_high, C2)
R2 = A*R1
C1 = C(fc_low, A, R1)

print(f"R1 = {R1/1e3:.2f}k, R2 = {R2/1e3:.2f}k")
print(f"C1 = {C1*1e6:.2f}u, C2 = {C2*1e9:.2f}n")