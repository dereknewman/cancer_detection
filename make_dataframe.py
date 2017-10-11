import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

random.seed(1321)
numpy.random.seed(1321)


xml_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_xml/"
xml_path = xml_dir + "185_088.xml"

def find_mhd_file(patient_id):
    """ find the '.mhd' file associated with a specific patient_id
    """
    src_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_mhd/"
    for src_path in glob.glob(src_dir + "*.mhd"):
        if patient_id in src_path:
            return src_path
    return None


def load_lidc_xml(xml_path):
    """ Read the xml file and create a csv with the (x,y,z) location, diameter, and malignacy of
    the positive examples, and a csv with the negative examples
    """
    #Empty dataframe
    full_df = pandas.DataFrame(columns=["patient_id", "x_center", "y_center", 
                                        "z_center", "diameter", "x_center_perc", 
                                        "y_center_perc", "z_center_perc", "diameter_perc", 
                                        "malscore", "spiculation", "lobulation", "file_path","xml_path"])
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return full_df #Empty dataframe
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
        
    #find the associated '.mhd' file, or return (None, None, None)
    src_path = find_mhd_file(patient_id)
    if src_path is None:
        return full_df #Empty dataframe

    print(patient_id)
    ###########################################################################
    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    rescale = spacing / settings.TARGET_VOXEL_MM
    ###########################################################################
    
    lines = []
    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        # print("Sesion")
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            # print("  ", nodule.noduleID)
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue
            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            #sphericiy = nodule.characteristics.sphericity.text
            #margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            #texture = nodule.characteristics.texture.text
            #calcification = nodule.characteristics.calcification.text
            #internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            #subtlety = nodule.characteristics.subtlety.text

            line = [patient_id, x_center, y_center, z_center, diameter, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, spiculation, lobulation, src_path, xml_path]
            #extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            lines.append(line)
            #extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            z_center -= origin[2]
            z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter = 0
            diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            # print("Non nodule!", z_center)
            line = [patient_id, x_center, y_center, z_center, diameter, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0, 0, 0, src_path, xml_path]
            lines.append(line)

    full_df = pandas.DataFrame(lines, columns=["patient_id", "x_center", "y_center", "z_center", "diameter", "x_center_perc", "y_center_perc", "z_center_perc", "diameter_perc", "malscore", "spiculation", "lobulation", "file_path","xml_path"])
    return full_df


xml_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_xml/"
full_dataframe = pandas.DataFrame(columns=["patient_id", "x_center", "y_center", 
                                    "z_center", "diameter", "x_center_perc", 
                                    "y_center_perc", "z_center_perc", "diameter_perc", 
                                    "malscore", "spiculation", "lobulation", "file_path", "xml_path"])

# Verify that the directory exists
if not os.path.isdir(xml_dir):
    print(xml_dir + " directory does not exist")

for file_ in os.listdir(xml_dir):
    # Verify that the file is a '.xml' file
    if not file_[-4:] == '.xml':
        continue
    df_temp = load_lidc_xml(xml_dir + file_)
    full_dataframe = full_dataframe.append(df_temp, ignore_index=True) #append to full
    
#full_dataframe.round({"coord_x": 4, "coord_y": 4, "coord_z":4})
full_dataframe = full_dataframe.drop_duplicates() #drop duplicate rows
full_dataframe.to_csv(settings.BASE_DIR + "patID_x_y_z_mal.csv", index=False)
    














