import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import Hopfieldnetwork
import os
import random
# This code what mostly taken from GitHub repository https://github.com/takyamamoto/Hopfield-Network but 
# We altered parts of the code such that it would be a better fit for our model

# Importing file
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

# Specify the relative paths to the extracted files
train_images_path = os.path.join(os.getcwd(), 'train-images.idx3-ubyte')
train_labels_path = os.path.join(os.getcwd(), 'train-labels.idx1-ubyte')
test_images_path = os.path.join(os.getcwd(), 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(os.getcwd(), 't10k-labels.idx1-ubyte')

# Load the training and testing images and labels
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(15, 15)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    # Setting up the image diagram. Taking the layout that was in the GitHub and altering it to contain more images
    fig, axarr = plt.subplots(len(test), 3, figsize=figsize)
    for ax_row in axarr:
        for ax in ax_row:
            ax.figure.set_size_inches(5, 5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  
    for i in range(len(test)):
        if i == 0:
            axarr[i, 0].set_title("Train data")
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')
        if i in range(0, 15): 
            axarr[i, 0].imshow(data[0])
            axarr[i, 0].axis('off')
        elif i in range(15, 30): 
            axarr[i, 0].imshow(data[1])
            axarr[i, 0].axis('off')
        elif i in range(30, 45): 
            axarr[i, 0].imshow(data[2])
            axarr[i, 0].axis('off')

    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

def preprocessing(img):
    w, h = img.shape
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data (60,000 test images)
    x_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    data = []

    # Define lists for selected data
    for digit in range(3):
        xi = x_train[y_train == digit]
        data.append(xi[0])

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]
    
    # Create Hopfield Network Model
    model = Hopfieldnetwork.HopfieldNetwork()
    model.train_weights(data)
    
    # Make test datalist (15 test images per digit)
    temp_test = []
    test = []

    
    for i in range(3):
        # Filter x_train based on y_train labels equal to the current digit
        xi = x_train[y_train == i]
        random.shuffle(xi)
        selected_samples = xi[1:16]        
        temp_test.append(selected_samples)

    for arr in temp_test:
        for img in arr:
            preprocessed_img = preprocessing(img)
            # Decreased inhibition (increase input corruption)
            # Implimented code to increase the corruption of the input image after the model is trained
            corruption_probability = 0.15
            corrupted_bits = np.random.rand(*img.shape) < corruption_probability
            noisy_image = img * (1 - corrupted_bits) + (1 - img) * corrupted_bits
            test.append(preprocessing(noisy_image)) 
    
    predicted = model.predict(test, threshold=75, asyn=False)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(10, 10))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()