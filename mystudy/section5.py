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

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out
    
    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx,dy

class AddLayer:
    def __init__(self):
        pass

    def foward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy

apple = 100
applenum = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple,applenum)
price = mul_tax_layer.forward(apple_price,tax)

print(price)

dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple,dapple_num,dtax)

