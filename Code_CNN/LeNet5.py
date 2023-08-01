import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import h5py
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Softmax
from tensorflow.keras.datasets import mnist
from PIL import Image

#get data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = (x_train)/255, (x_test)/255

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# padding
x_train_pad = tf.pad(x_train, [[0, 0], [2, 2], [2, 2],[0,0]], mode="CONSTANT")
x_test_pad = tf.pad(x_test, [[0, 0], [2, 2], [2, 2],[0,0]], mode="CONSTANT")

# Creating model
model = tf.keras.Sequential([
Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)),
AveragePooling2D(pool_size=(2,2), strides=2, padding="valid"),
Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
AveragePooling2D(pool_size=(2,2), strides=2, padding="valid"),
Flatten(),
Dense(units=120, activation='relu'),
Dense(units=84, activation='relu'),
Dense(units=10, activation='softmax'),
])


# Compilation
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

# Training
model.fit(
   x_train_pad,
   y_train,
   epochs=10,
    #validation_data=ds_test,
    #callbacks=[tensorboard_callback]
)

#model.load_weights(r'C:\Users\weights.h5')

model.summary()

model.evaluate(x_test_pad,y_test)

#model.save('C:/home/model.h5', save_format='h5')