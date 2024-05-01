Project.py
def reshape() and def plot(): Using the uploaded dataset, this part of the code is reshaping the dataset such that the dimensions match and that each element corresponds to each pixel on the image. The data, test, and predicted arrays are then converted to a 2 by 2 array and is used to plot the original training data (data), test data (test), and predicted output data (predicted). These two functions will be used later when the model is trained and help with the visualization of the hopfield neural network.
def preprocessing(): The preprocessing portion of the code is taking in the image provided and calculating the mean threshold value of the input image. It then returns an output which is used as the input to the Hopfield Network.
def main(): The main function uses the hopfield network and dataset, using the plot function to It visualizes the original training data, test data, and predicted output data and also to visualizes the weights of the Hopfield Network.
For project.py, For each digit, we are going to train the model with 10 images from the MNIST dataset and use this model to then test with 15 images and measure the accuracy.

Hopfieldnetwork.py
def train_weights(): Takes the training data and initializes the weights matrix W with a size of the number of neurons in the network set to zero. It then calculates the average of the activation level and implements Hebbian learning to update the weight matrix. The matrix is then normalized by making the diagonals zero to prevent self connections. The output of this function will produce an output from the hopfield network.
def _run(): This function is used in predict() to update the network's state such that it returns the final state of the network after the specified number of iterations.
def predict(): This function helps predict the output of the Hopfield network given input data. To do so, it uses the input data and uses the _run() method to predict the output state of the network.
def energy(): Calculates the energy of the network's state

bar_graph.py 
code using Final_Project_Data.csv to plot a bar graph with SEM error bars. We plotted a third degree polynomial and measured the Rsq values. 

Scatter_plot.py
Code uses Scatter_plot_data.csv to plot a scatter plot that and the curve of best fit. We also calculated the Rsq values. 

3d_plot.py
code using 3D_plot_data.csv to plot a 3D scatter graphs. We plotted a fitted plane and also calculated the Rsq values.  

results_mnist.png has the image of the training, input and output images. 

weights.png has the image of weight matrix of the 784 neurons showing connections between neurons in the model.