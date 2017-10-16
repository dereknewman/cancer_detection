#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:35:08 2017

@author: derek
"""

import os
import pandas as pd
import numpy as np
import glob
import ntpath
import SimpleITK
import helpers

BASE_DIR = "/media/derek/disk1/kaggle_ndsb2017/"
BASE_DIR_SSD = "/media/derek/disk1/kaggle_ndsb2017/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
TARGET_VOXEL_MM = 0.682
LUNA_SUBSET_START_INDEX = 0
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna_raw/"

def find_mhd_file(patient_id):
    """ find the '.mhd' file associated with a specific patient_id
    """
    for subject_no in range(LUNA_SUBSET_START_INDEX, 10):
        src_dir = LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None

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

def fetch_image(src_path):
    """Load the '.mhd' file, extract the 3D numpy array, rescale the data, 
    and normalize.
    """
    patient_id = ntpath.basename(src_path).replace(".mhd", "") #extract patient id from filename
    print("Patient: ", patient_id)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, TARGET_VOXEL_MM)
    return normalize(img_array)

src_dir = LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
# Verify that the directory exists
if not os.path.isdir(src_dir):
    print(src_dir + " directory does not exist")

full_dataframe = pd.DataFrame(columns=["patient_id", "coord_x", "coord_y", "coord_z", "malscore"]) 

for file_ in os.listdir(src_dir):
    # Verify that the file is a '.csv' file
    if not file_[-4:] == '.csv':
        continue
    df = pd.read_csv(src_dir + file_)
    patient_id = file_.split("_annos_")[0] #extrect the paitent id from filename
    patient_column = np.repeat(patient_id,df.shape[0])
    df_short = df[["coord_x", "coord_y", "coord_z", "malscore"]] #extract jus x,y,z and malscore
    df_short = df_short.assign(patient_id = patient_column) #add patient_id
    full_dataframe = full_dataframe.append(df_short, ignore_index=True) #append to full
    
#full_dataframe.round({"coord_x": 4, "coord_y": 4, "coord_z":4})
full_dataframe = full_dataframe.drop_duplicates() #drop duplicate rows
full_dataframe.to_csv(BASE_DIR + "patID_x_y_z_mal.csv", index=False)
    
#################


patients = full_dataframe.patient_id.unique()

for patient in patients[0:1]:
    patient_path = find_mhd_file(patient) #locate the path to the '.mhd' file
    image_array = fetch_image(patient_path)
    image_shape = image_array.shape
    print(patient)
    patient_df = full_dataframe.loc[full_dataframe['patient_id'] == patient] #create a dateframe assoicated to a single patient
    for index, row in patient_df.iterrows():
        x = int(row["coord_x"]*image_shape[0])
        y = int(row["coord_y"]*image_shape[1])
        z = int(row["coord_z"]*image_shape[2])
        
        print(x,y,z)
    
    
import matplotlib.pylab as plt
import matplotlib.patches as patches
fig1,ax1 = plt.imshow(image_array[z,:,:],cmap="gray")
rect = patches.Rectangle((x-32,y-32),x+32,y+32,linewidth=1,edgecolor='r',facecolor='none')
fig1.add_patch(rect)

plt.imshow(image_array[184,:,:],cmap="gray")

im = image_array[44,:,:]
image_ = np.stack([im, im, im],axis=2)
image_[x-32:x+32,y-32:y+32,0] = 30
plt.imshow(image_, cmap = "gray")



