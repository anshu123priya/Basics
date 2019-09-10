import keras
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
nb_classes = 10
nb_epoch=5
batch_size=32
img_rows, img_cols = 28,28

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
kernel_size=7
nb_filters=32
input_shape=(img_rows, img_cols,1)


model = Sequential()
model.add(Conv2D(nb_filters, kernel_size, strides=(1,1), padding='valid', input_shape = input_shape, activation='softmax'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid', data_format='channels_last'))
model.add(Flatten())
model.add(Dense(1024, activation='softmax'))
model.add(Dense(512, activation='softmax'))
model.add(Dense(256, activation='softmax'))
model.add(Dense(300, activation='softmax'))
model.add(Dense(10, activation='softmax'))
  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=5, verbose=2, validation_data=(x_test,y_test))
print(y_test[1:10])
