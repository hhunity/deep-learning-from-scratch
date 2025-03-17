import numpy as np
import matplotlib.pylab as plt
import cv2

def reLu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_function(x):
    y= x >0
    return y.astype(np.int32)

# print(step_function(np.array([2,3,-1])).astype(np.bool_))

def s3_1():
    x = np.arange(-5.0,5.1,0.1)

    y0= sigmoid(x)
    y1= step_function(x)
    y2= reLu(x)
    plt.plot(x,y0)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.ylim(-0.1,1.1)
    plt.show()

def s3_2():
    t = np.array([[1,2],[1,2]])
    x = np.arange(4).reshape((2,2))
    print(np.dot(t,x))
    print(t*x)



def s3_3():
    x = np.array([1,2])
    w = np.array([[1,3,5],[2,4,6]])
    print(np.dot(x,w))

# s3_3()

def identity_function(x):
    return x

def s3_4():
    #入力層
    x = np.array([1,0.5])
    w = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    b = np.array([0.1,0.2,0.3])
    a = np.dot(x,w)+b
    Z1 = sigmoid(a)

    #1->2
    W2 = np.array([[0.1,0.4],[0.2,0.5],[0.4,0.5]])
    B2 = np.array([0.1,0.2])
    A2 = np.dot(Z1,W2)+B2
    Z2 = sigmoid(A2)

    #2->out
    W3 = np.array([[0.2,0.3],[0.2,0.4]])
    B3 = np.array([0.1,0.2])
    A3 = np.dot(Z2,W3)+B3
    Y  = identity_function(A3)
    return Y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a
# print(softmax(np.array([0.3,2.9,4.0])))


import sys,os
sys.path.append(os.pardir)
from mnist import load_mnist

def s3_5():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
    img = x_train[0]
    label = t_train[0]

    img = img.reshape(28,28)
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import pickle

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("/Users/hiroyukih/vscode/deeplerning/ゼロから作るDeeplerning1/sample_weight.pkl","rb") as f:
        network = pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y  = softmax(a3)
    return y

# x,t = get_data()
# network = init_network()

# batch_size = 100
# accuracy_cnt = 0
# for i in range(0,len(x),batch_size):
#     x_bach = x[i:i+batch_size]
#     y_bach = predict(network,x_bach)
#     p = np.argmax(y_bach,axis=1)
#     # print(p)
#     accuracy_cnt += np.sum(p==t[i:i+batch_size])

# print(f"Accuracy: {float(accuracy_cnt)/len(x)}")

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

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)- f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1 + x

def funcion_2(x:np.ndarray):
    return x[0]**2 + x[1]**2

x = np.arange(0.0,20.0,0.1)
y = funcion_2(x)
# plt.plot(x,y)
# plt.show()

def numerical_gradient(f,x):
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

def gradient_descent(f,int_x,lr=0.01,step_num=100):
    x = int_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x-= lr * grad

    return x

init_x = np.array([-3.0,4.0])
g = gradient_descent(funcion_2,init_x,lr=0.1)
print(g)