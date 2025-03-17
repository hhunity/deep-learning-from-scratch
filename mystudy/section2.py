import numpy as np

# 2-5-2
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.4
    t = np.sum(x*w) + b
    if t  <= 0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    t = np.sum(x*w) + b
    if t  <= 0:
        return 0
    else:
        return 1
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    t = np.sum(x*w) + b
    if t  <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1,s2)
    
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

# 2-3-2
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    t = np.sum(x*w) + b
    if t  <= 0:
        return 0
    else:
        return 1

# print(AND(0,1))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))

# 2-3-1
def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    t = x1*w1+x2*w2
    if t < theta:
        return 0
    else:
        return 1
    
# print(AND(0,1))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))