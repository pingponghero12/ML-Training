import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
tf.experimental.numpy.experimental_enable_numpy_behavior()



(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.reshape(train_images, (60000, 28, 28, 1)).astype('float32') / 255
test_images = tf.reshape(test_images, (10000, 28, 28, 1)).astype('float32') / 255

def one_layer_model(n, act_type):
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(n, activation=act_type))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    return test_acc

def two_layer_model(n1, n2, act_type):
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(n1, activation=act_type))
    model.add(layers.Dense(n2, activation=act_type))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    return test_acc


def one_conv_model(n):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(n, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    return test_acc

one_layer_input = np.array([4, 6, 8, 10, 16, 32, 64, 128, 256, 512])
one_layer_acc_relu = np.empty(shape=(one_layer_input.size, 3))
one_layer_acc_softmax = np.empty(shape=(one_layer_input.size, 3))

for idx, i in enumerate(one_layer_input):
    for j in range(3):
        one_layer_acc_relu[idx][j] = one_layer_model(i, 'relu')

for idx, i in enumerate(one_layer_input):
    for j in range(3):
        one_layer_acc_softmax[idx][j] = one_layer_model(i, 'softmax')

fig = plt.figure()

y1 = np.mean(one_layer_acc_relu, axis=1)
y2 = np.mean(one_layer_acc_softmax, axis=1)

yerr1_lower = np.abs(y1 - np.min(one_layer_acc_relu, axis=1))
yerr1_higher = np.abs(y1 - np.min(one_layer_acc_relu, axis=1))
yerr1 = np.array([yerr1_lower, yerr1_higher])

yerr2_lower = np.abs(y2 - np.min(one_layer_acc_softmax, axis=1))
yerr2_higher = np.abs(y2 - np.min(one_layer_acc_softmax, axis=1))
yerr2 = np.array([yerr2_lower, yerr2_higher])


plt.errorbar(one_layer_input, y1, yerr=yerr1, fmt='o-', label='ReLU')
plt.errorbar(one_layer_input, y2, yerr=yerr2, fmt='o-', label='Softmax')

plt.xlabel('Input Size')
plt.ylabel('Accuracy')
plt.title('One Layer Model: ReLU vs Softmax')
plt.grid()
plt.legend()

fig.savefig("one_layer_acc_relu.png")
plt.show()


# Flatten the 2D arrays
relu_flat = one_layer_acc_relu.flatten()
softmax_flat = one_layer_acc_softmax.flatten()

# Create the savedata array
savedata = np.column_stack((np.repeat(one_layer_input, 3), relu_flat, softmax_flat))

# Create a header that matches the number of columns
header = 'neuron_number,relu,softmax'

# Save to CSV
np.savetxt('output.csv', savedata, delimiter=',', header=header, comments='')

