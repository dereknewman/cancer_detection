#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:51:14 2017

@author: derek
"""
import pandas as pd
import ntpath
import SimpleITK
import numpy as np
import settings
import helpers
import tensorflow as tf

TARGET_VOXEL_MM = 0.682


def normalize(image):
    """ Normalize image -> clip data between -1000 and 400. Scale values to 0 to 1. #### SCALE TO -.5 to .5 ##### TODO:???????
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def plot_slice_box(image_array,z_perc,y_perc,x_perc):
    import matplotlib.pylab as plt
    import cv2
    image_array = image_array - image_array.min()
    image_array = (image_array/image_array.max())*255
    z_,y_,x_ = image_array.shape
    z_loc = int(round(z_*z_perc))
    y_loc = int(round(y_*y_perc))
    x_loc = int(round(x_*x_perc))
    im_slice = image_array[z_loc,:,:].copy()
    im_slice[y_loc-16:y_loc+16,x_loc-16] = 255
    im_slice[y_loc-16:y_loc+16,x_loc+16] = 255
    im_slice[y_loc-16,x_loc-16:x_loc+16] = 255
    im_slice[y_loc+16,x_loc-16:x_loc+16] = 255
    #plt.imshow(im_slice,cmap="gray")
    cv2.imwrite(settings.BASE_DIR + "/resources/_images/img_" + str(z_loc)+'_'+str(y_loc)+'_'+str(x_loc) + ".png", im_slice)

def extract_cube(image_array,z_perc,y_perc,x_perc):
    im_z, im_y, im_x = image_array.shape
    z_min = int(round(z_perc*im_z)) - 16
    y_min = int(round(y_perc*im_y)) - 16
    x_min = int(round(x_perc*im_x)) - 16
    z_max = int(round(z_perc*im_z)) + 16
    y_max = int(round(y_perc*im_y)) + 16
    x_max = int(round(x_perc*im_x)) + 16
    if z_min < 0:
        z_max = z_max + abs(z_min)
        z_min = 0
    if y_min < 0:
        y_max = y_max + abs(y_min)
        y_min = 0
    if x_min < 0:
        x_max = x_max + abs(x_min)
        x_min = 0
    if z_max > im_z:
        z_min = z_min - abs(z_max - im_z)
        z_max = im_z
    if y_max > im_y:
        y_min = y_min - abs(y_max - im_y)
        y_max = im_y
    if x_max > im_x:
        x_min = x_min - abs(x_max - im_x)
        x_max = im_x
    image_cube = image_array[z_min:z_max,y_min:y_max,x_min:x_max]
    return image_cube

def patient_to_tfrecord(save_path,patient_id, image_array, patient_df):
    patient_id = "1.4.5.6.123551485448654"
    tfrecord_file = patient_id + ".tfrecord"  
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(1000):
        add_to_tfrecord(writer,image_cube, label)
    writer.close()

def add_to_tfrecord(writer,image_cube, label):
        image_cube = np.asarray(image_cube,np.int16) #ensure data is in int16
        binary_cube = image_cube.tobytes()
        image_label = np.array(label,np.int16) #ensure data is in int16
        binary_label = image_label.tobytes()
        shape = np.array(image_cube.shape, np.int32) #ensure data is in int16
        binary_shape = shape.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_label])),
                'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_shape])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
                }))
        writer.write(example.SerializeToString())



#################
save_path = "/media/derek/disk1/kaggle_ndsb2017/resources/_cubes/"
full_dataframe = pd.read_csv(settings.BASE_DIR + "patID_x_y_z_mal.csv")
patients = full_dataframe.patient_id.unique()
#patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.131939324905446238286154504249"
for patient in patients[2:5]:
    patient_df = full_dataframe.loc[full_dataframe['patient_id'] == patient] #create a dateframe assoicated to a single patient
    patient_df = patient_df.sort_values('z_center')
    patient_path = patient_df.file_path.unique()[0]  #locate the path to the '.mhd' file
    
    ####image_array = fetch_image(patient_path)
    ####image_shape = image_array.shape
    print(patient)
    
    #####################################
    #### Load and process image  ########
    #####################################
    itk_img = SimpleITK.ReadImage(patient_path)
    if (np.array(itk_img.GetDirection()) != np.array([ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.])).all():
        print("WARNING!!!!! Image in different direction")
    image_array = SimpleITK.GetArrayFromImage(itk_img)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    image_array = helpers.rescale_patient_images(image_array, spacing, TARGET_VOXEL_MM)
    #Do inline normalization (i.e. NOT HERE)
    #image_array = normalize(image_array)
    
    #image_cubes = []
    #image_labels = []
    tfrecord_file = save_path + patient + ".tfrecord" 
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for index, row in patient_df.iterrows():
        #TEMP#####
        z_perc = row["z_center_perc"]
        y_perc = row["y_center_perc"]
        x_perc = row["x_center_perc"]
        #plot_slice_box(image_array,z_perc,y_perc,x_perc)
        image_cube = extract_cube(image_array,z_perc,y_perc,x_perc)
        image_label = (row["malscore"], row["spiculation"], row["lobulation"])
        add_to_tfrecord(writer,image_cube, image_label)
        #image_cubes.append(image_cube)
        #image_labels.append(image_label)
        #TEMP#####
    writer.close()
    #np.save(settings.BASE_DIR + "resources/_cubes/" + patient + '_train.npy', (image_cubes, image_labels))
