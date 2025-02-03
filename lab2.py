import numpy as np
import matplotlib.pyplot as plt

'''
Legg inn kode for å importere data fra csv
Bruker midlertidige dummy arrays
'''

x = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
y = [0, 0, 0, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]

# Calculate the cross-correlation of x and y
cross_correlation = np.correlate(x, y, mode='full')

# Calculate the autocorrelation of x
auto_correlation = np.correlate(x, x, mode="same")

# Plot both the cross-correlation and the autocorrelation
fig, ax = plt.subplots(2)
ax[0].stem(auto_correlation)
ax[0].set_title('Auto-correlation of x')
ax[1].stem(cross_correlation)
ax[1].set_title('Cross-correlation of x and y')
plt.tight_layout()
plt.show()