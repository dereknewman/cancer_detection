#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:51:14 2017

@author: derek
"""
import pandas as pd
import SimpleITK
import numpy as np
import tensorflow as tf
import cv2

TARGET_VOXEL_MM = 0.682
BASE_DIR = "/media/derek/disk1/kaggle_ndsb2017/"

def normalize(image):
    """ Normalize image -> clip data between -1000 and 400. Scale values to -0.5 to 0.5 
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image -= 0.5
    return image

def plot_slice_box(image_array,z_perc,y_perc,x_perc):
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
    cv2.imwrite(BASE_DIR + "/resources/_images/img_" + str(z_loc)+'_'+str(y_loc)+'_'+str(x_loc) + ".png", im_slice)

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

def add_to_tfrecord(writer,image_cube, label):
        image_cube = np.asarray(image_cube,np.int16) #ensure data is in int16
        image_shape = image_cube.shape
        binary_cube = image_cube.tobytes()
        binary_label = np.array(image_label, np.int16).tobytes()
        binary_shape = np.array(image_shape, np.int16).tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_shape])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_label])),
                'cube': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
                }))
        writer.write(example.SerializeToString())


def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False, verbose=False):
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = np.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res

#################
save_path = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/"
full_dataframe = pd.read_csv(BASE_DIR + "patID_x_y_z_mal.csv")
patients = full_dataframe.patient_id.unique()
#patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.131939324905446238286154504249"
for patient in patients:
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
    image_array = rescale_patient_images(image_array, spacing, TARGET_VOXEL_MM)
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
