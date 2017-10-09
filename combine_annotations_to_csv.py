#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:35:08 2017

@author: derek
"""
import settings
import os
import pandas as pd
import numpy as np
import glob
import ntpath
import SimpleITK
import helpers
import cv2

def find_mhd_file(patient_id):
    """ find the '.mhd' file associated with a specific patient_id
    """
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None

def normalize(image):
    """ Normalize image -> clip data between -1000 and 400. Scale values to 0 to 1. #### SCALE TO -.5 to .5 ##### TODO:???????
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
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
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)
    return normalize(img_array)

src_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
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
full_dataframe.to_csv(settings.BASE_DIR + "patID_x_y_z_mal.csv", index=False)
    
#################


patients = full_dataframe.patient_id.unique()

for patient in patients:
    patient_path = find_mhd_file(patient) #locate the path to the '.mhd' file
    image_array = fetch_image(patient_path)
    print(patient)
    patient_df = full_dataframe.loc[full_dataframe['patient_id'] == patient] #create a dateframe assoicated to a single patient
    fetch_image(src_path)
    for index, row in patient_df.iterrows():
        print(row)
        
    










