import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, models


tf.experimental.numpy.experimental_enable_numpy_behavior()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.reshape(train_images, (60000, 28, 28, 1)).astype('float32') / 255
test_images = tf.reshape(test_images, (10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}")
print(f"Test acc: {test_acc}")
