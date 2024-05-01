import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Reading the csv file and converting the data to an array
csv_file = '3D_plot_data.csv'
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        data.append([float(val) for val in row])

data = np.array(data)

# Obtaining x, y, z variables from the data
x_values = data[:, 0]
y_values = data[:, 1]
z_values = data[:, 2]

# Model function for the plane
def plane_model(xy, a, b, c):
    x, y = xy
    return a * x + b * y + c

# Multiplanar fitting
popt, _ = curve_fit(plane_model, (x_values, y_values), z_values)

# Generate points for the fitted plane and plot the plane
x_plane = np.linspace(min(x_values), max(x_values), 10)
y_plane = np.linspace(min(y_values), max(y_values), 10)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = plane_model((X_plane, Y_plane), *popt)

# Calculate Rsq value
r_sq = 1 - np.sum((z_values - plane_model((x_values, y_values), *popt))**2) / np.sum((z_values - np.mean(z_values))**2)

# Equation of the plane
equation = f"z = {popt[0]:.5f} * x + {popt[1]:.5f} * y + {popt[2]:.5f}"
print("Equation of the plane:", equation)

# Create the scatter and fitted curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_values, y_values, z_values)

ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, color='lightblue')

ax.set_xlabel('Number of Neurons Deleted')
ax.set_ylabel('Percent Corruption (%)')
ax.set_zlabel('Percent Accuracy (%)')
plt.title(f'3D Scatter Plot with Fitted Plane, $R^2 = {r_sq:.5f}$')

plt.show()
