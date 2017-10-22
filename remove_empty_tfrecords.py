#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:27:08 2017

@author: derek
"""

import os
import numpy as np
import shutil


cubes_path = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/"
tf_list = os.listdir(cubes_path)
train_perc = .8

for f in tf_list:
    if os.stat(cubes_path + f).st_size == 0:
        os.remove(cubes_path + f)

if not os.path.exists(cubes_path + "label0"):
    os.makedirs(cubes_path + "label0")
if not os.path.exists(cubes_path + "label1"):
    os.makedirs(cubes_path + "label1")
if not os.path.exists(cubes_path + "label2"):
    os.makedirs(cubes_path + "label2")
if not os.path.exists(cubes_path + "label3"):
    os.makedirs(cubes_path + "label3")
if not os.path.exists(cubes_path + "label4"):
    os.makedirs(cubes_path + "label4")
if not os.path.exists(cubes_path + "label5"):
    os.makedirs(cubes_path + "label5")    


tf_list = os.listdir(cubes_path)
for f in tf_list:
    f_split = f.split("_")
    if len(f_split) == 2:
        if f_split[1][0] == "0":
            os.rename(cubes_path + f, cubes_path + "label0/" + f)
        if f_split[1][0] == "1":
            os.rename(cubes_path + f, cubes_path + "label1/" + f)
        if f_split[1][0] == "2":
            os.rename(cubes_path + f, cubes_path + "label2/" + f)
        if f_split[1][0] == "3":
            os.rename(cubes_path + f, cubes_path + "label3/" + f)
        if f_split[1][0] == "4":
            os.rename(cubes_path + f, cubes_path + "label4/" + f)
        if f_split[1][0] == "5":
            os.rename(cubes_path + f, cubes_path + "label5/" + f)

tf_list = os.listdir(cubes_path)
total_size = 0
max_size = 0
dir_sizes = {}
for label_ in tf_list:
    if os.path.isdir(cubes_path + label_): 
        label_path = cubes_path + label_ + "/"
        dir_size = sum(os.path.getsize(label_path + f) for f in os.listdir(label_path) if os.path.isfile(label_path + f))
        dir_sizes[label_path]=dir_size
        total_size += dir_size
        if dir_size > max_size:
            max_size = dir_size

#add integer number of repeats of files to each directory to get close to evening out sizes
tf_list = os.listdir(cubes_path) 
for label_ in tf_list:
    if os.path.isdir(cubes_path + label_): 
        label_path = cubes_path + label_ + "/"
        label_files = os.listdir(label_path)
        file_sizes = [os.path.getsize(label_path + f) for f in label_files if os.path.isfile(label_path + f)]
        dir_size = np.sum(file_sizes)
        files = [f for f in label_files if os.path.isfile(label_path + f)]
        #running_sum = np.cumsum(file_sizes)
        add_size = max_size - dir_size
        label_files = os.listdir(label_path)
        duplicate_index = int(float(add_size)/dir_size)
        duplicate_list = files*(duplicate_index-1)
        for ff in duplicate_list:
            shutil.copyfile(label_path + ff, label_path + ff[:-9] +str(np.random.randint(10000,99999)) + ff[-9:])
            
#recalculate max_size
tf_list = os.listdir(cubes_path)
total_size = 0
max_size = 0
dir_sizes = {}
for label_ in tf_list:
    if os.path.isdir(cubes_path + label_): 
        label_path = cubes_path + label_ + "/"
        dir_size = sum(os.path.getsize(label_path + f) for f in os.listdir(label_path) if os.path.isfile(label_path + f))
        dir_sizes[label_path]=dir_size
        total_size += dir_size
        if dir_size > max_size:
            max_size = dir_size

#add remaining files to even out difference
tf_list = os.listdir(cubes_path) 
for label_ in tf_list:
    if os.path.isdir(cubes_path + label_): 
        label_path = cubes_path + label_ + "/"
        label_files = os.listdir(label_path)
        file_sizes = [os.path.getsize(label_path + f) for f in label_files if os.path.isfile(label_path + f)]
        dir_size = np.sum(file_sizes)
        files = np.array([f for f in label_files if os.path.isfile(label_path + f)])
        running_sum = np.cumsum(file_sizes)
        add_diff = max_size - dir_size
        copy_list = files[running_sum < add_diff]
        for ff in copy_list:
            shutil.copyfile(label_path + ff, label_path + ff[:-9] +str(np.random.randint(10000,99999)) + ff[-9:])


#move files to train and test folders
tf_list = os.listdir(cubes_path) 
if not os.path.exists(cubes_path + "train"):
    os.makedirs(cubes_path + "train")
if not os.path.exists(cubes_path + "test"):
    os.makedirs(cubes_path + "test")    
for label_ in tf_list:
    if os.path.isdir(cubes_path + label_): 
        label_path = cubes_path + label_ + "/"
        label_files = os.listdir(label_path)
        label_files.sort()
        label_files_len = len(label_files)
        train_len = int(label_files_len * train_perc)
        for ff in label_files[:train_len]:
             os.rename(label_path + ff, cubes_path + "train/" + ff)
        for ff in label_files[train_len:]:
             os.rename(label_path + ff, cubes_path + "test/" + ff)

       
        
        
        