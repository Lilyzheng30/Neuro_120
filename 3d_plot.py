import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read data from CSV file
csv_file = '3D_plot_data.csv'
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        data.append([float(val) for val in row])

data = np.array(data)

# Extract x, y, z coordinates from the data
x_values = data[:, 0]
y_values = data[:, 1]
z_values = data[:, 2]

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_values, y_values, z_values)

ax.set_xlabel('Number of Neurons Deleted')
ax.set_ylabel('Percent Corruption (%)')
ax.set_zlabel('Percent Accuracy (%)')

plt.title('3D Scatter Plot')
plt.show()
