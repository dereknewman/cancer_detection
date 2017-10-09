import os
#COMPUTER_NAME = os.environ['COMPUTERNAME']
#print("Computer: ", COMPUTER_NAME)

TARGET_VOXEL_MM = 0.70
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "/media/derek/disk1/kaggle_ndsb2017/"
BASE_DIR = "/media/derek/disk1/kaggle_ndsb2017/"
EXTRA_DATA_DIR = "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna_raw/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"

