import struct
import numpy as np
import matplotlib.pyplot as plt

train_imgs_filename = 'MNIST/train-images-idx3-ubyte'
train_lbls_filename = 'MNIST/train-labels-idx1-ubyte'
train_imgs_file = open(train_imgs_filename, 'rb')
train_lbls_file = open(train_lbls_filename, 'rb')

#read headers
train_imgs_header = struct.iter_unpack('>i', train_imgs_file.read(16))
imgs_magic = next(train_imgs_header)[0]
imgs = next(train_imgs_header)[0]
rows = next(train_imgs_header)[0]
cols = next(train_imgs_header)[0]

digits = 10

train_lbls_header = struct.iter_unpack('>i', train_lbls_file.read(8))
lbls_magic = next(train_lbls_header)[0]
lbls = next(train_lbls_header)[0]

print("%d, %d, %d" % (imgs, rows, cols))

#read training imgs into np array
train_imgs = np.zeros((imgs, rows, cols)) #square format
train_input_layers = np.zeros((imgs, rows * cols)) #layer format
print("reading in images...")
for i in range(imgs):
    train_img = train_imgs_file.read(rows*cols)
    train_imgs[i] = np.array([int(j) for j in train_img]).reshape(rows,cols)
    train_input_layers[i] = np.array([int(j) for j in train_img])

#read labels into np array
train_lbls = np.array([int(i) for i in train_lbls_file.read(imgs)]) #single digit
train_truth_layers = np.zeros((imgs, digits)) #expected layers
for i in range(imgs):
    train_truth_layers[i][train_lbls[i]] = 1


'''
We now have our database fully read in. Our images are a 3d numpy array and our labels are a 1d numpy array.
Now we want to create a network object that can be trained.
By trained, what we mean is adjust weights and biases over several epochs.
'''

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

class Net: # accepts a tuple indicating the number of nodes in each layer. contains the weights array and biases vector for each layer.

    def __init__(self, layers): # must have atleast 2 layers (2 items in the layers tuple)
        self.layers = layers #tuple
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.zeros((layers[i],layers[i-1])))
        self.biases = []
        for i in range(1, len(layers)):
            self.biases.append(np.zeros((layers[i])))
        self.net = [] # contains activations
        for i in range(len(self.layers)):
            self.net.append(np.zeros((self.layers[i])))

    def dump(self):
        print(self.layers)
        print(self.biases)
        print(self.weights)

    def forward(self, input_layer):
        self.net[0] = input_layer
        for i in range(1, len(net)):
            self.net[i] = sigmoid(np.dot(self.weights[i-1], self.net[i-1]) + self.biases[i-1])
        return self.net[len(self.net)-1]

    def ssd(self, expected, actual): #sum of squared differences
        squared_diff = np.square(expected-actual)
        return np.sum(squared_diff)

    def loss(self, labels, inputs):
        ssd_array = np.zeros((len(inputs)))
        for i in range(len(inputs)):
            output = self.forward(inputs[i])
            ssd_array[i] = self.ssd(labels[i], output)
        return np.mean(ssd_array)

    def backprop(self, label, example): # for one training example
        d_weights = self.weights
        d_biases = self.biases
        d_activations = self.net

        cost = np.square(self.net[last] - expected)
        d_activations[len(self.layers)] = cost

        for i in reversed(range(len(self.layers)-1)):
            for j in range(len(self.net[i])):
                for k in range(len(self.net[i+1])):
                    d_activation[
        return d_weights, d_biases

        

    def train(self, data, labels, batch_size): #labels and data are arrays of vectors (or 2d arrays), 1 epoch
        np.random.shuffle(data)
        batches = np.zeros((floor(len(data)/batch_size), batch_size, rows*cols))
        for i in range(floor(len(data)/batch_size)):
            for j in range(batch_size):
                batches[i][j] = data[i*batch_size + j]

        for i in range(len(batches)):
            d_weights_list = []
            d_biases_list = []
            for j in range(len(batches[i]):
                this_d_weights, this_d_biases = backprop(batches[i][j])
                d_weights_list.append(this_d_weights)
                d_biases_list.append(this_d_biases)
            d_weights = np.mean(d_weights_list, 0)
            d_biases = np.mean(d_biases_list, 0)
            self.weights -= d_weights
            self.biases -= d_biases
    
testnet = Net((rows*cols,16,16,10))
print(testnet.loss(train_truth_layers, train_input_layers))
#testnet.dump()
#print(testnet.forward([]))
