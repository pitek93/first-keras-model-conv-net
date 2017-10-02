from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import struct
from random import randint

def read_network_dataset(dataset):
    fname_img = '/home/user/emnist-letters-test-images-idx3-ubyte'
    fname_lbl = '/home/user/emnist-letters-test-labels-idx1-ubyte'
    # wczytanie etykiet
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    # wczytanie obrazków
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    return lbl,img

testing_labels, testing_pixels = read_network_dataset('testing')
testing_pixels = testing_pixels.reshape(testing_pixels.shape[0], 28, 28, 1)


a = randint(0, testing_pixels.shape[0])
print(a)

X = testing_pixels[a, :, :, 0]
plt.imshow(X)
plt.show()

model = load_model('cnn_letters_v1.h5')

print(model.predict(X.reshape(1, 28, 28, 1)))