import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import Hopfieldnetwork


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

# Specify the paths to the extracted files
train_images_path = '/Users/lilyzheng/Documents/GitHub/Neuro_120/train-images.idx3-ubyte'
train_labels_path = '/Users/lilyzheng/Documents/GitHub/Neuro_120/train-labels.idx1-ubyte'
test_images_path = '/Users/lilyzheng/Documents/GitHub/Neuro_120/t10k-images.idx3-ubyte'
test_labels_path = '/Users/lilyzheng/Documents/GitHub/Neuro_120/t10k-labels.idx1-ubyte'

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

def plot(data, test, predicted, figsize=(5, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(test), 3, figsize=figsize)
    for i in range(len(test)):
        if i==0:
            axarr[i, 0].set_title("Train data")
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
            
        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')
            
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
    print(len(data))
    data = [preprocessing(d) for d in data]
    
    # Create Hopfield Network Model
    model = Hopfieldnetwork.HopfieldNetwork()
    model.train_weights(data)
    
    # Make test datalist (5 test images, one per digit)
    test = []
    for i in range(3):
        xi = x_train[y_train==i]
        test.append(xi[1])
    test = [preprocessing(d) for d in test]
    
    predicted = model.predict(test, threshold=50, asyn=False)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()