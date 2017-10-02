import numpy as np
import struct

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D


def read_network_dataset(dataset):
    if dataset is "training":
        fname_img = '/home/user/emnist-letters-train-images-idx3-ubyte'
        fname_lbl = '/home/user/emnist-letters-train-labels-idx1-ubyte'
    elif dataset is "testing":
        fname_img = '/home/user/emnist-letters-test-images-idx3-ubyte'
        fname_lbl = '/home/user/emnist-letters-test-labels-idx1-ubyte'
    else:
        print("ERROR")
        return
    # loading labels
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    # loading pictures
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    return lbl,img

# training data
training_labels, training_pixels = read_network_dataset('training')
print(training_labels.shape)
print(training_pixels.shape)

# testing data
testing_labels, testing_pixels = read_network_dataset('testing')
print(testing_labels.shape)
print(testing_pixels.shape)

# reshape for keras input
training_pixels = training_pixels.reshape(training_pixels.shape[0], 28, 28, 1)
testing_pixels = testing_pixels.reshape(testing_pixels.shape[0], 28, 28, 1)

training_pixels = training_pixels.astype('float32')
testing_pixels = testing_pixels.astype('float32')
training_pixels /= 255
testing_pixels /= 255

training_labels = np_utils.to_categorical(training_labels)
testing_labels = np_utils.to_categorical(testing_labels)

# model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', ))  # 1st layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # 2nd layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling layer
model.add(Dropout(0.25))  # dropout ro prevent overfitting (dropping 25% neurons)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(27, activation='linear'))

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# network training
model.fit(training_pixels, training_labels,
          batch_size=32, epochs=10, verbose=1)

# model saving
model.save('cnn_letters_v1.h5')

# network accuracy testing
score = model.evaluate(testing_pixels, testing_labels, verbose=1)

print(' Loss:', score[0] * 100)
print(' Acc:', score[1] * 100)
