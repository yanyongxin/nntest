import struct
import numpy as np
import matplotlib.pyplot as plt

train_imgs_filename = 'MNIST/train-images-idx3-ubyte'
train_imgs_file = open(train_imgs_filename, 'rb')

#read header

train_imgs_header = struct.iter_unpack('>i', train_imgs_file.read(16))
magic = next(train_imgs_header)[0]
imgs = next(train_imgs_header)[0]
rows = next(train_imgs_header)[0]
cols = next(train_imgs_header)[0]

print("%d, %d, %d, %d" % (magic, imgs, rows, cols))

#read one image into a numpy array

train_img = train_imgs_file.read(rows*cols) 
X = np.array([int(i) for i in train_img]).reshape(rows,cols)
plt.imshow(X, 'gray')

plt.show()
