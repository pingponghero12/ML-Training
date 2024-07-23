import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

a = tf.constant([[1, 2], [3, 4]])
print(a)


b = np.array([[2, 2], [2, 2]])
print("b: ", b)

print("a*b: ", a*b)
