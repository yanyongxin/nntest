import math
import struct
import sys
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

#read training imgs into np array
train_imgs = np.zeros((imgs, rows, cols)) #square format
train_input_layers = np.zeros((imgs, rows * cols)) #layer format
print("reading in images...")
for i in range(imgs):

    train_img = train_imgs_file.read(rows*cols)
    train_imgs[i] = np.array([int(j) for j in train_img]).reshape(rows,cols)
    train_input_layers[i] = np.array([int(j) for j in train_img])/255

#read labels into np array
train_lbls = np.array([int(i) for i in train_lbls_file.read(imgs)]) #single digit
train_truth_layers = np.zeros((imgs, digits)) #expected layers
for i in range(imgs):
    train_truth_layers[i][train_lbls[i]] = 1

#plt.imshow(train_imgs[0])
#plt.show()
#np.set_printoptions(threshold=sys.maxsize)

'''
We now have our database fully read in. Our images are a 3d numpy array and our labels are a 1d numpy array.
Now we want to create a network object that can be trained.
By trained, what we mean is adjust weights and biases over several epochs.
'''

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def d_sigmoid(x):
    return(np.exp(x)/np.square(np.exp(x)+1))

class Net: # accepts a tuple indicating the number of nodes in each layer. contains the weights array and biases vector for each layer

    def __init__(self, layers): # must have atleast 2 layers (2 items in the layers tuple)
        self.layers = layers #tuple
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.full((layers[i],layers[i-1]), 0.01))
        self.biases = []
        for i in range(1, len(layers)):
            self.biases.append(np.full((layers[i]), 0.001))
        self.net = [] # contains activations
        for i in range(len(layers)):
            self.net.append(np.zeros((layers[i])))

    def dump(self):
        print("DUMP========")
        print(self.biases)
        print(self.weights)

    def forward(self, input_layer):
        self.net[0] = input_layer
        for i in range(1, len(self.net)):
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

    def d_cost(self, expected, actual):
        return 2 * (expected - actual)

    def backprop(self, label, example): # for one training example
        self.forward(example)
        d_weights = []
        for i in range(len(self.layers)-1):
            d_weights.append(np.zeros((self.layers[i+1],self.layers[i])))
        d_biases = []
        for i in range(len(self.layers)-1):
            d_biases.append(np.zeros((self.layers[i+1])))
        d_activations = [] 
        for i in range(len(self.layers)):
            d_activations.append(np.zeros((self.layers[i])))

        #d_cost = 2 * (self.net[len(self.layers)-1] - label)
        d_activations[len(self.layers)-1] = self.d_cost(self.net[len(self.layers)-1], label)

        for i in reversed(range(len(self.layers)-1)):
            for j in range(len(self.net[i])):
                for k in range(len(self.net[i+1])):
                    #calculates z for the whole layer (layer i)
                    z_layer = np.dot(self.weights[i][k], self.net[i]) + self.biases[i][k]
                    #calculates z for this specific node (layer i, index j)
                    z = self.weights[i][k][j] * self.net[i][j] + self.biases[i][k]
                    #calculates the suggested change 
                    d_activations[i][j] += (self.weights[i][k][j] * d_sigmoid(z_layer) * d_activations[i+1][k])/len(self.net[i+1])
                    d_biases[i][k] += (d_sigmoid(z_layer) * d_activations[i+1][k])/len(self.net[i])
                    d_weights[i][k][j] = self.net[i][j] * d_sigmoid(z) * d_activations[i+1][k]
        return d_weights, d_biases

        

    def train(self, data, labels, batch_size, *, batches=0): #labels and data are arrays of vectors (or 2d arrays), 1 epoch
        np.random.shuffle(data)
        data_batched = np.zeros((math.floor(len(data)/batch_size), batch_size, rows*cols))
        labels_batched = np.zeros((math.floor(len(data)/batch_size), batch_size, digits))
        for i in range(math.floor(len(data)/batch_size)):
            for j in range(batch_size):
                data_batched[i][j] = data[i*batch_size + j]
                labels_batched[i][j] = labels[i*batch_size + j]
        
        batch_count = batches
        if (batches == 0):
            batch_count = len(data_batched)
        for i in range(batch_count):
            print("\n\nbatch", i)
            d_weights_list = []
            d_biases_list = []
            d_weights = []
            for i in range(1, len(self.layers)):
                d_weights.append(np.zeros((self.layers[i],self.layers[i-1])))
            d_biases = []
            for i in range(len(self.layers)-1):
                d_biases.append(np.zeros((self.layers[i+1])))
            for j in range(len(data_batched[i])):
                this_d_weights, this_d_biases = self.backprop(labels_batched[i][j], data_batched[i][j])
                d_weights_list.append(this_d_weights)
                d_biases_list.append(this_d_biases)
            for j in range(len(self.layers)-1):    
                for k in range(len(data_batched[i])):
                    d_weights[j] += d_weights_list[k][j]
                    d_biases[j] += d_biases_list[k][j]
                d_weights[j] = d_weights[j]/len(data_batched[j])
                d_biases[j] = d_biases[j]/len(data_batched[j])
                print("[A]", self.weights[j])
                self.weights[j] = np.subtract(self.weights[j], d_weights[j])
                self.biases[j] = np.subtract(self.biases[j], d_biases[j])
                print("[B]", self.weights[j])
            #print(self.forward(train_input_layers[5000]))
            #print(train_truth_layers[5000])
            #print(self.forward(train_input_layers[9000]))
            #print(train_truth_layers[9000])
            print("loss:", self.loss(train_truth_layers, train_input_layers))
print("create net")
testnet = Net((rows*cols, 16, 16, 10))
print("net created, calculating initial loss")
print(testnet.loss(train_truth_layers, train_input_layers))
testnet.dump()
'''
plt.imshow(train_imgs[0])
plt.show()
plt.imshow(train_imgs[1])
plt.show()
plt.imshow(train_imgs[2])
plt.show()
print(testnet.forward(train_input_layers[0]))
print(testnet.forward(train_input_layers[1]))
print(testnet.forward(train_input_layers[2]))
'''
print("train for one epoch")
testnet.train(train_input_layers, train_truth_layers, 10)
print("done. recalculating loss:")
print(testnet.loss(train_truth_layers, train_input_layers))
#print(testnet.forward([]))

