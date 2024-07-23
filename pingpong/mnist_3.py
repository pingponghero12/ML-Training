import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, models

tf.experimental.numpy.experimental_enable_numpy_behavior()



(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.reshape(train_images, (60000, 28, 28, 1)).astype('float32') / 255
test_images = tf.reshape(test_images, (10000, 28, 28, 1)).astype('float32') / 255

def one_layer_model(n):
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(n, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    return test_acc

# Tests for one_layer_model(128)
# Acc1:  0.968999981880188
# Acc2:  0.9740999937057495
# Acc3:  0.975600004196167
# Acc4:  0.9732999801635742
# Acc5:  0.9731000065803528

print("Acc1: ", one_layer_model(128))
print("Acc2: ", one_layer_model(128))
print("Acc3: ", one_layer_model(128))

