from utilitites import *

save = True
window = True
f_min=-500
f_max=500

measurements = ["slow", "fast", "reverse"]
# Create 2D matrix with velocity estimates for each measurement
estimates = np.zeros((len(measurements), 4))

for i, m in enumerate(measurements):
    print(f"Measurement: {m}")
    for j in range(4):
        filename = f"radar_{m}_{j+1}.csv"
        estimates[i][j] = radar_all(filename, f_min, f_max, save=save, window=window)
       
    # Calculate mean and std for each measurement
    print("--------------------------------")
    print(f"Mean: {np.mean(estimates[i]):.2f}")
    print(f"Std: {np.std(estimates[i]):.2f}")
    print("---------------------------------------------------")