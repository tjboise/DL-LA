import numpy as np
import matplotlib.pyplot as plt

# Load the data from the sensi.dat file
sensitivity_data = np.fromfile("sensi.dat", dtype=np.float64)

# Print the loaded data
print(sensitivity_data)

plt.plot(sensitivity_data)
plt.show()




