import numpy as np

# Målinger fra pulssensoren (Rød, Grønn, Blå)
R = np.array([65.3, 84.3, 67.1, 62.0, 66.1])
G = np.array([68.0, 87.2, 71.2, 90.9, 58.6])
B = np.array([58.2, 60.1, 60.5, 86.6, 58.9])

# Referanseverdier (fra sportsklokken)
ref = np.array([64, 63, 63.5, 65, 66.5])


std = np.array([np.std(R), np.std(G), np.std(B)])
mean = np.array([np.mean(R), np.mean(G), np.mean(B)])

print(f"Mean: {mean}")
print(f"Std: {std}")

err = np.zeros((3, 5))
for i in range(5):
    err[0][i] = abs(R[i] - ref[i])
    err[1][i] = abs(G[i] - ref[i])
    err[2][i] = abs(B[i] - ref[i])

avg_err = np.zeros(3)
for i in range(3):
    avg_err[i] = np.mean(err[i])
    print(f"Avg error for {i}: {avg_err[i]}")

