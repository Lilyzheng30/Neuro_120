import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Scatter plot that consist of a fitted curve and the Rsq value
def scatter(csv_file, x_values, y_values, x_header, y_header, plot_title):
    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)
   
    fitted_curve = np.poly1d(np.polyfit(x_values, y_values, 3))

    # Solving for Rsq value 
    y_fitted = fitted_curve(x_values)
    r_sq = 1 - np.sum((y_values - y_fitted)**2) / np.sum((y_values - np.mean(y_values))**2)

    equation = f"y = {fitted_curve[3]:.7f}x^3 + {fitted_curve[2]:.5f}x^2 + {fitted_curve[1]:.5f}x + {fitted_curve[0]:.5f}"
    print("Equation of the curve:", equation)
    
    plt.scatter(x_values, y_values)
    x_range = np.linspace(min(x_values), max(x_values), 100)
    plt.plot(x_range, fitted_curve(x_range), color='lightblue', linestyle='--', label=f'Fitted Curve, $R^2 = {r_sq:.5f}$')

    plt.xlabel(x_header)
    plt.ylabel(y_header)  
    plt.title(plot_title)
    plt.xticks(x_values)
    plt.grid(False)
    plt.legend()
    plt.show()

# Import csv
csv_file = 'Scatter_plot_data.csv'
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    data = list(zip(*reader))

# Plot for Percent Accuracy vs Percent Corruption
scatter(csv_file, data[0], [float(val) for val in data[1]], headers[0], headers[1], 'Percent Accuracy vs Percent Corruption')
# Plot for Percent Accuracy vs Number of Ablated Neurons
scatter(csv_file, data[2], [float(val) for val in data[3]], headers[2], headers[3], 'Percent Accuracy vs Number of Ablated Neurons')