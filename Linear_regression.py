#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/4 11:11
# @Author  : Little Feng
# @Email   : LittleFeng@email.com
# @File    : Linear_regression.py
# @Software: PyCharm

def computing_error(initial_k,initial_b,data_samples):
    totalerror = 0
    num = data_samples.shape[1]
    for i in range(num):
        x = data_samples[0,i]
        y = data_samples[1,i]
        totalerror += (y - initial_k*x - initial_b)**2
    return totalerror/num

def gradient_descent(k,b,learning_rate,data_samples):
    deta_gradient_k = 0
    deta_gradient_b = 0
    num = data_samples.shape[1]
    for i in range(num):
        x = data_samples[0, i]
        y = data_samples[1, i]
        deta_gradient_k += -(2/num) * x * (k*x+b-y)
        deta_gradient_b += -(2/num) * (k*x+b-y)
    k += learning_rate * deta_gradient_k
    b += learning_rate * deta_gradient_b
    return [k, b]



def gradient_descent_iter(initial_k,initial_b,
                              learning_rate,data_samples,num_iterations):
    k = initial_k
    b = initial_b
    for i in range(num_iterations):
        k,b = gradient_descent(k,b,learning_rate, data_samples)
    return [k, b]


import numpy as np
from matplotlib import pyplot as plt
if __name__ == '__main__':
    x_data = np.linspace(0,10,50)
    y_data = 2 * x_data + 1 + np.random.normal(0,0.4,50)
    data = np.array([x_data,y_data])
    np.save('data.npy',data)
    data_samples = np.load('data.npy')

    learning_rate = 10**(-4)
    initial_k = 0
    initial_b = 0
    num_iterations = pow(10,4)
    print('initial parameters k = {0}, b = {1} and '
          'the error = {2}'.format(initial_k,initial_b,
                             computing_error(initial_k,initial_b
                                             ,data_samples)))
    [k, b] = gradient_descent_iter(initial_k,initial_b,
                              learning_rate,data_samples,num_iterations)
    print('the learning result k = {0}, b = {1} and '
          'the error = {2}'.format(k,b, computing_error(
        k, b, data_samples
    )))

    plt.scatter(x_data,y_data)
    x_1 = np.linspace(0,10,10)
    y_1 = k * x_1 + b
    plt.plot(x_1,y_1,'r')
    plt.show()






