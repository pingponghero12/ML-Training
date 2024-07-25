import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

tf.experimental.numpy.experimental_enable_numpy_behavior()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images =train_images.reshape((60000, 784)).T.astype('float32') / 255
test_images = test_images.reshape((10000, 784)).T.astype('float32') / 255

def init_params():
    W1 = np.random.randn(10, 784) 
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) 
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def RELU(Z):
    return np.maximum(0,Z)
    
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def forward_propag(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = RELU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLu(Z):
    return Z > 0

def back_propag(A1,A2,Z1,Z2,W2,X, Y):
    m = Y.size
    one_hot_Y=one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m* dZ2.dot(A1.T)
    db2 = 1/m* np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2)* deriv_ReLu(Z1)
    dW1 = 1/m* dZ1.dot(X.T)
    db1 = 1/m* np.sum(dZ1, axis=1, keepdims=True) 
    return dW1,  db1,dW2 , db2

def uptd_proc(W1, W2,b1,b2,db1,db2, dW1,dW2,alpha):
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    return W1,b1,W2,b2

def predict(A2):
    return np.argmax(A2,0)

def acc(predict, Y):
    
    return np.mean(predict == Y)


def gradient_descent(X,Y,iteration,alpha):
    W1,b1,W2,b2=init_params()
    for i in range(iteration):
        Z1,A1,Z2,A2 = forward_propag(W1,b1,W2,b2,X)
        dW1 , db1 , dW2,db2 = back_propag(A1,A2,Z1,Z2,W2,X,Y)
        W1 , b1 , W2,b2= uptd_proc(W1,W2,b1,b2,db1 ,db2,dW1,dW2,alpha)
        if (i%10==0):
            print("iteracje",i)
            print("Acc:", acc(predict(A2),Y))
    return W1,b1,W2,b2

W1,b1,W2,b2 = gradient_descent(train_images[:, :2000],train_labels[:2000], 2000,0.1)
