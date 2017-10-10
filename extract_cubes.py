#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:51:14 2017

@author: derek
"""
import pandas as pd

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


#################
full_dataframe = pd.read_csv(settings.BASE_DIR + "patID_x_y_z_mal.csv")
patients = full_dataframe.patient_id.unique()

for patient in patients[0:1]:
    print("A")
    patient_path = full_dataframe[full_dataframe.patient_id == patient].file_path.unique()[0]  #locate the path to the '.mhd' file
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