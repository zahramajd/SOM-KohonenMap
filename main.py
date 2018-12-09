import scipy.io
import math
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import array


def euclidean_distance(x1,x2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x2)]))

def my_plot(training_data,units):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = []
    y = []
    z = []

    for i in range(0,len(training_data)):
        x.append(training_data[i,0])
        y.append(training_data[i,1])
        z.append(training_data[i,2])

    ax.scatter3D(x, y, z,color='red')

    x1 = []
    y1 = []
    z1 = []
    for i in range(0,len(units)):
        x1.append(units[i,0])
        y1.append(units[i,1])
        z1.append(units[i,2])

    ax.scatter3D(x1, y1, z1,color='blue')

    plt.show()
    
    return 

def point_diff(x1,x2):
    return [(x1[0]-x2[0]),(x1[1]-x2[1]),(x1[2]-x2[2])]

def point_sum(x1,x2):
    return [(x1[0]+x2[0]),(x1[1]+x2[1]),(x1[2]+x2[2])]

def point_multiply_constant(point,value):
    return [value*point[0],value*point[1],value*point[2]]
    
def gaussian_function(distance, sigma):
    return np.exp(-0.5 * ((math.pow(distance,2))/math.pow(sigma,2)))


# load data
mat = scipy.io.loadmat('Chainlink.mat')
data=mat['Chainlink']

# split into train & test
training_data = data[0:800]
test_data = data[801:]
units = []

# generate random weights
for i in range(0, 100):
    units.append([random(),random(),random()])
units = array(units)


training_cycle = 1700

# learning rate parameters
learning_rate_0 = 0.1
learning_rate_constant = 1000


sigma_0 = 0.0741
# sigma_constant = 1000 / (math.log(sigma_0))
sigma_constant = 394.852

for i in range(0, training_cycle):
    learning_rate = learning_rate_0 * np.exp((-1 * i)/learning_rate_constant)
    sigma = sigma_0 * np.exp((-1 * i)/sigma_constant)

    for j in range(0, len(training_data)):
        for k in range(0,len(units)):
            distance = euclidean_distance(training_data[j],units[k])
            units[k]= point_sum(units[k],point_multiply_constant(point_diff(training_data[j],units[k]),
            learning_rate * gaussian_function(distance,sigma)))

my_plot(training_data, units)
