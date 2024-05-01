import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def scatter(csv_file, x_values, y_values, x_header, y_header, plot_title):
    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)
   
    fitted_curve = np.poly1d(np.polyfit(x_values, y_values, 3))
    
    y_fitted = fitted_curve(x_values)
    r_sq = 1 - np.sum((y_values - y_fitted)**2) / np.sum((y_values - np.mean(y_values))**2)
    
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

csv_file = 'Scatter_plot_data.csv'
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    data = list(zip(*reader))

scatter(csv_file, data[0], [float(val) for val in data[1]], headers[0], headers[1], 'Percent Accuracy vs Percent Corruption')
scatter(csv_file, data[2], [float(val) for val in data[3]], headers[2], headers[3], 'Percent Accuracy vs Number of Ablated Neurons')