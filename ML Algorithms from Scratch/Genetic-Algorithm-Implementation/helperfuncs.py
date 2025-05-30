import numpy as np
import matplotlib.pyplot as plt  
import math

def rastriginsfunc(inputs):
    dimension      =  len(inputs)
    output         =  np.sum(inputs**2-10*np.cos(2*math.pi*inputs))+10*dimension
    return output

def ackleyfunc(inputs):
    dimension      =  len(inputs)
    output         =  -20*np.exp(-.2*np.sqrt(np.sum(inputs**2)/dimension))-np.exp(np.sum(np.cos(2*math.pi*inputs))/dimension)+20+np.exp(1)
    return output

def bentcigarfunc(inputs):
    output         = 0
    output         = (inputs[0]**2 ) + (10**6)*(np.sum( inputs[ 1: ]**2))
    return output

def spherefunc(inputs):
    output         = np.sum(inputs**2)
    return output


def func(type,inputs):
    if type == "sphere":
        return spherefunc(inputs)
    elif type == "ackley":
        return ackleyfunc(inputs)
    elif type == "bentcigar":
        return bentcigarfunc(inputs)
    elif type == "rastrigins":
        return rastriginsfunc(inputs)


def plot_result(inputs):
    plt.plot(inputs , 'r')
    plt.xlabel('iteration') 
    plt.ylabel('cost') 
    plt.title('cost in every iteration') 
    plt.show()