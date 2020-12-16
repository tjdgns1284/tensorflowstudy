import numpy as np 
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return list(map(lambda x: math.tanh(x),x ))


def relu(x):
    result = []
    for ele in x:
        if ele<=0:
            result.append(0)
        else: 
            result.append(ele)
    return result

x = np.linspace(-4,4,100)
sig = sigmoid(x)
r = relu(x)
ta= tanh(x)
plt.plot(x,r)
plt.show()