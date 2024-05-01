import csv
import numpy as np
import matplotlib.pyplot as plt

#calculated SEM for Error bars
def calculate_sem(data):
    return np.std(data) / np.sqrt(len(data))

def plotgraph(csv_file, x_values, y_values, x_header, y_header, plot_title):
    x_pos = np.arange(len(x_values))
    sem_values = [calculate_sem(y_values)] * len(y_values)

    plt.bar(x_pos, y_values, yerr=sem_values, capsize=5, width=0.5, color='lightblue')
    plt.xlabel(x_header)
    plt.ylabel(y_header)  
    plt.title(plot_title)
    plt.xticks(x_pos, x_values)
    plt.grid(False)
    plt.show()

csv_file = 'Final_Project_Data.csv'
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    data = list(zip(*reader))

plotgraph(csv_file, data[0], [float(val) for val in data[1]], headers[0], headers[1], 'Percent Accuracy vs Percent Corruption')
plotgraph(csv_file, data[2], [float(val) for val in data[3]], headers[2], headers[3], 'Percent Accuracy vs Number of Ablated Neurons')
