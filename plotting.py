#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:16:20 2017

@author: derek
"""

import numpy as np
import matplotlib.pylab as plt


def plot_data(f_name):
    f = open(f_name)
    data = []
    for l in f:
        data.append(float(l))
    f.close()
    data = np.array(data)
    
    avg_num = 10
    data_avg = np.reshape(data[:(len(data)//avg_num)*avg_num],(-1,avg_num))
    data_avg = np.mean(data_avg,axis=1)
    data_avg = np.repeat(data_avg,avg_num,axis=0)
    data_avg = data_avg.reshape((-1,avg_num))
    data_avg = data_avg.flatten()
    plt.figure()
    plt.plot(data)
    plt.plot(data_avg)


f_path = "/media/derek/disk1/kaggle_ndsb2017/"
f_name_test = "test3_values_rand_1e-05.txt"
f_name_train = "train3_values_rand_1e-05.txt"

plot_data(f_path + f_name_train)
plot_data(f_path + f_name_test)


