import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
import cv2
from mnist import load_mnist

def reLu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_function(x):
    y= x >0
    return y.astype(np.int32)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a
# print(softmax(np.array([0.3,2.9,4.0])))


def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y))/batch_size

def cross_entropy_error2(y,t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y[np.arange(batch_size),t]))/batch_size

def numerical_gradient1D(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1   = f(x)

        x[idx] = tmp_val - h
        fxh2   = f(x)

        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val
    
    return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def gradient_descent(f,int_x,lr=0.01,step_num=100):
    x = int_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x-= lr * grad

    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

# net = simpleNet()
# print(net.W)
# x = np.array([0.6,0.9])
# p = net.predict(x)
# np.argmax(p)
# r = net.loss(x,np.array([0,0,1]))
# print(r)

# dW = numerical_gradient(f,net.W)

x = np.arange(100).reshape([10,-1])
print(x)
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    idx = it.multi_index
    print(idx)
    it.iternext()